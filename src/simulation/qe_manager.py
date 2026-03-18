"""
qe_manager.py - Quantum ESPRESSO interface for DFT validation

PURPOSE:
    After ML screening identifies promising candidates, we validate
    the top 10-20 with full DFT calculations using Quantum ESPRESSO (QE).
    This module generates QE input files, runs calculations, and parses output.

WHAT IS DFT?
    Density Functional Theory - a quantum mechanical method that computes
    the ground-state electronic structure of materials.
    - Input: crystal structure (atom types, positions, cell)
    - Output: total energy, forces, electronic band structure, DOS
    - Accuracy: ~1-10 meV/atom (much better than ML potentials)
    - Cost: hours to days per structure (vs seconds for ML)

WHAT IS QUANTUM ESPRESSO?
    Free, open-source DFT code. Widely used in materials science.
    Uses plane-wave basis set + pseudopotentials.
    Key executables: pw.x (SCF/relax), bands.x, dos.x, ph.x (phonons)

ALGORITHM:
    1. Generate input file (pw.x format) from ASE Atoms object
    2. Set calculation parameters (cutoffs, k-points, convergence)
    3. Run pw.x (can take hours for a single structure)
    4. Parse output for energy, forces, electronic structure
    5. For TE properties: run band structure → feed to BoltzTraP2

INSTALLATION:
    On Windows: Use WSL2 (Windows Subsystem for Linux)
    1. Enable WSL2: wsl --install
    2. Install Ubuntu from Microsoft Store
    3. In Ubuntu: sudo apt install quantum-espresso
    OR compile from source for GPU support:
    https://www.quantum-espresso.org/Doc/user_guide/node14.html

PSEUDOPOTENTIALS:
    QE needs pseudopotentials (PP) for each element.
    Free PP libraries:
    - SSSP (Standard Solid-State Pseudopotentials): recommended
    - PSlibrary
    Download from: https://www.materialscloud.org/discover/sssp/table/efficiency
"""

from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from ase import Atoms
    from ase.io import write as ase_write
    from ase.calculators.espresso import Espresso, EspressoProfile
except ImportError:
    Atoms = None
    logger.warning("ASE espresso interface not available")


class QEManager:
    """
    Manages Quantum ESPRESSO calculations for DFT validation.

    HOW IT WORKS:
        1. Takes a relaxed structure from ML screening
        2. Generates QE input file with appropriate parameters
        3. Optionally runs the calculation (if QE is installed)
        4. Parses the output

    NOTE: QE runs in WSL2 on Windows. This module generates input files
    that you can manually run in WSL2 if automatic execution doesn't work.

    TYPICAL CALCULATION SEQUENCE:
        1. SCF (self-consistent field) → ground state energy
        2. Relax (optional) → DFT-optimized structure
        3. NSCF (non-self-consistent) on dense k-grid → for DOS/bands
        4. Bands calculation → electronic band structure
        5. DOS calculation → density of states
        6. BoltzTraP2 → thermoelectric properties from bands
    """

    def __init__(
        self,
        qe_path: str = "",
        pseudo_dir: str = "data/pseudopotentials",
        output_dir: str = "data/results/dft",
        ecutwfc: float = 60.0,
        ecutrho: float = 480.0,
        kpoints: tuple = (8, 8, 1),
        vacuum: float = 20.0,
    ):
        self.qe_path = qe_path
        self.pseudo_dir = Path(pseudo_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        self.kpoints = kpoints
        self.vacuum = vacuum

        # Pseudopotential file mapping (SSSP efficiency library)
        # You need to download these files
        self.pseudopotentials = {
            "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "V": "V.pbe-spnl-kjpaw_psl.1.0.0.UPF",
            "Cr": "Cr.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Nb": "Nb.pbe-spn-kjpaw_psl.0.3.0.UPF",
            "Mo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Zr": "Zr.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Hf": "Hf.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Ta": "Ta.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "W": "W.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Sc": "Sc.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
            "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
            "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
            "F": "F.pbe-n-kjpaw_psl.1.0.0.UPF",
            "Cl": "Cl.pbe-n-kjpaw_psl.1.0.0.UPF",
            "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF",
            "Se": "Se.pbe-n-kjpaw_psl.1.0.0.UPF",
            "Te": "Te.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "H": "H.pbe-kjpaw_psl.1.0.0.UPF",
            "Br": "Br.pbe-n-kjpaw_psl.1.0.0.UPF",
        }

    def generate_scf_input(self, atoms: "Atoms", name: str, calculation: str = "scf") -> Path:
        """
        Generate Quantum ESPRESSO SCF input file.

        QE INPUT FILE FORMAT (pw.x):
            &CONTROL
                calculation = 'scf'       ! type: scf, relax, vc-relax, bands, nscf
                prefix = 'name'           ! output file prefix
                outdir = './tmp'          ! scratch directory
                pseudo_dir = './pseudo'   ! pseudopotential directory
            /
            &SYSTEM
                ibrav = 0                 ! 0 = cell vectors given explicitly
                nat = N                   ! number of atoms
                ntyp = M                  ! number of atom types
                ecutwfc = 60              ! kinetic energy cutoff (Ry)
                ecutrho = 480             ! charge density cutoff (Ry)
                occupations = 'smearing'  ! for metals/small-gap
                smearing = 'mp'           ! Marzari-Vanderbilt cold smearing
                degauss = 0.02            ! smearing width (Ry)
            /
            &ELECTRONS
                conv_thr = 1e-8           ! SCF convergence threshold (Ry)
                mixing_beta = 0.3         ! charge mixing (slower but stable)
            /
            ATOMIC_SPECIES
                Ti 47.867 Ti.pbe-spn-kjpaw_psl.1.0.0.UPF
                C  12.011 C.pbe-n-kjpaw_psl.1.0.0.UPF
                ...
            CELL_PARAMETERS angstrom
                a1x a1y a1z
                a2x a2y a2z
                a3x a3y a3z
            ATOMIC_POSITIONS angstrom
                Ti x y z
                C  x y z
                ...
            K_POINTS automatic
                kx ky kz 0 0 0

        PARAMETERS EXPLAINED:
            ecutwfc: Plane-wave cutoff energy. Higher = more accurate but slower.
                     60 Ry is standard for PAW pseudopotentials.
            ecutrho: Charge density cutoff. Usually 8-12x ecutwfc for PAW.
            k-points: Sampling of reciprocal space. More = more accurate.
                      (8,8,1) for 2D slabs (1 in z-direction = vacuum).
            smearing: For metallic systems, we smear the Fermi surface.
                      'mp' (Marzari-Vanderbilt) is good for metals.
            conv_thr: SCF convergence. 1e-8 Ry is standard for good accuracy.
        """
        if Atoms is None:
            raise ImportError("ASE required")

        calc_dir = self.output_dir / name
        calc_dir.mkdir(parents=True, exist_ok=True)

        # Get unique elements
        symbols = list(set(atoms.get_chemical_symbols()))
        symbols.sort()

        # Build input file manually for full control
        lines = []

        # CONTROL namelist
        lines.append("&CONTROL")
        lines.append(f"  calculation = '{calculation}'")
        lines.append(f"  prefix = '{name}'")
        lines.append(f"  outdir = './tmp'")
        lines.append(f"  pseudo_dir = '{self.pseudo_dir}'")
        if calculation in ("relax", "vc-relax"):
            lines.append(f"  forc_conv_thr = 1.0d-4")
        lines.append("/")

        # SYSTEM namelist
        lines.append("&SYSTEM")
        lines.append(f"  ibrav = 0")
        lines.append(f"  nat = {len(atoms)}")
        lines.append(f"  ntyp = {len(symbols)}")
        lines.append(f"  ecutwfc = {self.ecutwfc}")
        lines.append(f"  ecutrho = {self.ecutrho}")
        lines.append(f"  occupations = 'smearing'")
        lines.append(f"  smearing = 'mp'")
        lines.append(f"  degauss = 0.02")
        lines.append("/")

        # ELECTRONS
        lines.append("&ELECTRONS")
        lines.append(f"  conv_thr = 1.0d-8")
        lines.append(f"  mixing_beta = 0.3")
        lines.append("/")

        # IONS (for relax calculations)
        if calculation in ("relax", "vc-relax"):
            lines.append("&IONS")
            lines.append(f"  ion_dynamics = 'bfgs'")
            lines.append("/")

        # ATOMIC_SPECIES
        lines.append("ATOMIC_SPECIES")
        for sym in symbols:
            mass = atoms[atoms.get_chemical_symbols().index(sym)].mass
            pp = self.pseudopotentials.get(sym, f"{sym}.UPF")
            lines.append(f"  {sym} {mass:.3f} {pp}")

        # CELL_PARAMETERS
        cell = atoms.get_cell()
        lines.append("CELL_PARAMETERS angstrom")
        for vec in cell:
            lines.append(f"  {vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")

        # ATOMIC_POSITIONS
        positions = atoms.get_positions()
        chem_symbols = atoms.get_chemical_symbols()
        lines.append("ATOMIC_POSITIONS angstrom")
        for sym, pos in zip(chem_symbols, positions):
            lines.append(f"  {sym} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")

        # K_POINTS
        lines.append(f"K_POINTS automatic")
        kx, ky, kz = self.kpoints
        lines.append(f"  {kx} {ky} {kz} 0 0 0")

        # Write input file
        input_file = calc_dir / f"{name}.{calculation}.in"
        with open(input_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Generated QE input: {input_file}")

        # Also write a run script
        run_script = calc_dir / "run.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Run Quantum ESPRESSO SCF calculation for {name}\n")
            f.write(f"# Execute in WSL2: bash run.sh\n\n")
            f.write(f"mkdir -p tmp\n")
            f.write(f"mpirun -np 4 pw.x < {name}.{calculation}.in > {name}.{calculation}.out 2>&1\n")
            f.write(f"echo 'Calculation complete. Check {name}.{calculation}.out for results.'\n")

        return input_file

    def generate_bands_input(self, atoms: "Atoms", name: str) -> Path:
        """
        Generate input files for band structure calculation.

        SEQUENCE (3 calculations needed):
            1. SCF → self-consistent ground state (already done)
            2. NSCF on k-path → non-self-consistent on band path
            3. bands.x → extract eigenvalues

        K-PATH for hexagonal MXenes:
            Γ → M → K → Γ (standard hexagonal BZ path)
            Γ = (0, 0, 0)
            M = (0.5, 0, 0)
            K = (1/3, 1/3, 0)

        The band structure is needed for BoltzTraP2 to compute
        thermoelectric transport properties.
        """
        calc_dir = self.output_dir / name
        calc_dir.mkdir(parents=True, exist_ok=True)

        # K-path for hexagonal Brillouin zone
        # Each line: kx ky kz weight
        kpath = []
        n_points = 30  # points between high-symmetry points

        # Γ → M
        for i in range(n_points):
            t = i / n_points
            kpath.append(f"  {0.5*t:.6f} {0.0:.6f} {0.0:.6f} 1.0")

        # M → K
        for i in range(n_points):
            t = i / n_points
            kx = 0.5 - t * (0.5 - 1/3)
            ky = t * 1/3
            kpath.append(f"  {kx:.6f} {ky:.6f} {0.0:.6f} 1.0")

        # K → Γ
        for i in range(n_points + 1):
            t = i / n_points
            kx = 1/3 * (1 - t)
            ky = 1/3 * (1 - t)
            kpath.append(f"  {kx:.6f} {ky:.6f} {0.0:.6f} 1.0")

        # Generate NSCF input with k-path
        scf_input = self.generate_scf_input(atoms, name, calculation="scf")

        # Read the SCF input and modify for bands
        with open(scf_input) as f:
            content = f.read()

        bands_content = content.replace("calculation = 'scf'", "calculation = 'bands'")
        bands_content = bands_content.replace(
            f"K_POINTS automatic\n  {self.kpoints[0]} {self.kpoints[1]} {self.kpoints[2]} 0 0 0",
            f"K_POINTS crystal\n{len(kpath)}\n" + "\n".join(kpath)
        )

        bands_file = calc_dir / f"{name}.bands.in"
        with open(bands_file, "w") as f:
            f.write(bands_content)

        logger.info(f"Generated QE bands input: {bands_file}")
        return bands_file

    def parse_scf_output(self, output_file: str | Path) -> dict:
        """
        Parse QE SCF output file for key results.

        WHAT WE EXTRACT:
            - Total energy (Ry)
            - Fermi energy (eV)
            - Total force (Ry/Bohr)
            - Convergence status
            - Number of SCF iterations
            - Wall time
        """
        output_file = Path(output_file)
        if not output_file.exists():
            return {"error": f"File not found: {output_file}"}

        results = {
            "converged": False,
            "total_energy_ry": None,
            "total_energy_ev": None,
            "fermi_energy_ev": None,
            "total_force": None,
            "n_scf_steps": 0,
            "wall_time": None,
        }

        with open(output_file) as f:
            for line in f:
                if "!" in line and "total energy" in line:
                    # !    total energy              =    -234.56789 Ry
                    parts = line.split("=")
                    energy_ry = float(parts[-1].strip().split()[0])
                    results["total_energy_ry"] = energy_ry
                    results["total_energy_ev"] = energy_ry * 13.6057  # Ry → eV

                elif "the Fermi energy is" in line:
                    parts = line.split("is")
                    results["fermi_energy_ev"] = float(parts[-1].strip().split()[0])

                elif "Total force" in line:
                    parts = line.split("=")
                    results["total_force"] = float(parts[-1].strip().split()[0])

                elif "convergence has been achieved" in line:
                    results["converged"] = True
                    # Extract iteration count
                    parts = line.split("in")
                    if len(parts) > 1:
                        results["n_scf_steps"] = int(parts[-1].strip().split()[0])

                elif "PWSCF" in line and "WALL" in line:
                    results["wall_time"] = line.strip()

        return results
