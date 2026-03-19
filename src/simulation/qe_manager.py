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
        # Ensure forward slashes for WSL/Linux paths
        pseudo_path = str(self.pseudo_dir).replace("\\", "/")
        lines.append(f"  pseudo_dir = '{pseudo_path}'")
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

    def generate_nscf_input(self, atoms: "Atoms", name: str, kpoints: tuple = (16, 16, 1)) -> Path:
        """
        Generate NSCF (non-self-consistent field) input for dense k-grid.

        WHY NSCF?
            The SCF calculation converges the charge density on a coarse k-grid
            (e.g., 8x8x1). But for accurate DOS and transport properties, we need
            eigenvalues on a MUCH denser k-grid (16x16x1 or more).

            NSCF reads the converged charge density from SCF, then computes
            eigenvalues at many more k-points WITHOUT re-converging the density.
            This is much faster than doing SCF on the dense grid.

        ALGORITHM:
            1. Copy SCF input
            2. Change calculation = 'scf' -> 'nscf'
            3. Increase k-grid density
            4. Add nosym = .true. (needed for BoltzTraP2 compatibility)
            5. Increase nbnd (number of bands) for transport calculations

        PARAMETERS:
            kpoints: dense k-grid, default (16,16,1) for 2D MXenes
                     More k-points = better DOS resolution but slower
        """
        calc_dir = self.output_dir / name
        calc_dir.mkdir(parents=True, exist_ok=True)

        # Read SCF input as template
        scf_file = calc_dir / f"{name}.scf.in"
        if not scf_file.exists():
            # Generate SCF first
            self.generate_scf_input(atoms, name, calculation="scf")

        with open(scf_file) as f:
            content = f.read()

        # Modify for NSCF
        nscf_content = content.replace("calculation = 'scf'", "calculation = 'nscf'")

        # Add nosym and increase bands before the closing /SYSTEM
        # nosym = .true. is needed for BoltzTraP2 to work correctly
        nscf_content = nscf_content.replace(
            "  degauss = 0.02\n/",
            "  degauss = 0.02\n  nosym = .true.\n  nbnd = 40\n/"
        )

        # Replace k-grid with denser one
        old_kline = f"  {self.kpoints[0]} {self.kpoints[1]} {self.kpoints[2]} 0 0 0"
        new_kline = f"  {kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0"
        nscf_content = nscf_content.replace(old_kline, new_kline)

        nscf_file = calc_dir / f"{name}.nscf.in"
        with open(nscf_file, "w") as f:
            f.write(nscf_content)

        logger.info(f"Generated QE NSCF input: {nscf_file}")
        return nscf_file

    def generate_dos_input(self, name: str) -> Path:
        """
        Generate dos.x input file for density of states calculation.

        WHY DOS?
            The density of states tells us:
            - Whether the material is metallic (DOS at Fermi level > 0) or
              semiconducting (gap around Fermi level)
            - The bandgap magnitude (distance between VBM and CBM)
            - Shape of DOS near Fermi level (affects Seebeck coefficient)

            For thermoelectrics: sharp DOS features near the Fermi level
            lead to high Seebeck coefficients (Mahan-Sofo theory).

        ALGORITHM:
            dos.x reads the NSCF output and computes the DOS by:
            1. Collecting all eigenvalues from the dense k-grid
            2. Broadening each eigenvalue with a Gaussian/tetrahedron method
            3. Summing contributions to get total DOS(E)

        INPUT FORMAT:
            &DOS
                prefix = 'name'        ! must match NSCF prefix
                outdir = './tmp'       ! must match NSCF outdir
                fildos = 'name.dos'    ! output DOS file
                Emin = -10.0           ! energy range (eV relative to Fermi)
                Emax = 10.0
                DeltaE = 0.01          ! energy resolution (eV)
            /
        """
        calc_dir = self.output_dir / name
        calc_dir.mkdir(parents=True, exist_ok=True)

        lines = [
            "&DOS",
            f"  prefix = '{name}'",
            f"  outdir = './tmp'",
            f"  fildos = '{name}.dos'",
            f"  Emin = -10.0",
            f"  Emax = 10.0",
            f"  DeltaE = 0.01",
            "/",
        ]

        dos_file = calc_dir / f"{name}.dos.in"
        with open(dos_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Generated QE DOS input: {dos_file}")
        return dos_file

    def generate_bands_pp_input(self, name: str) -> Path:
        """
        Generate bands.x post-processing input.

        WHY bands.x?
            After pw.x bands calculation, the eigenvalues are stored in QE's
            internal binary format. bands.x extracts them into a text file
            that can be plotted or fed to BoltzTraP2.

        INPUT FORMAT:
            &BANDS
                prefix = 'name'
                outdir = './tmp'
                filband = 'name.bands.dat'  ! output band structure file
                lsym = .true.               ! symmetry analysis of bands
            /
        """
        calc_dir = self.output_dir / name
        calc_dir.mkdir(parents=True, exist_ok=True)

        lines = [
            "&BANDS",
            f"  prefix = '{name}'",
            f"  outdir = './tmp'",
            f"  filband = '{name}.bands.dat'",
            f"  lsym = .true.",
            "/",
        ]

        bands_pp_file = calc_dir / f"{name}.bands_pp.in"
        with open(bands_pp_file, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Generated QE bands.x input: {bands_pp_file}")
        return bands_pp_file

    def generate_master_script(self, names: list[str], nprocs: int = 4) -> Path:
        """
        Generate a master bash script that runs all DFT calculations sequentially.

        CALCULATION SEQUENCE PER CANDIDATE:
            1. pw.x SCF        (~10-60 min)  - ground state
            2. pw.x NSCF       (~5-30 min)   - dense k-grid eigenvalues
            3. dos.x           (~1 min)       - density of states
            4. pw.x bands      (~5-30 min)   - band structure along k-path
            5. bands.x         (~1 min)       - post-process bands

        TOTAL ESTIMATED TIME:
            ~30-120 min per candidate with 4 MPI processes on 16 threads.
            For 5 candidates: ~2.5-10 hours total.

        WSL2 PATH CONVERSION:
            Windows D:\\MXDiscovery -> WSL /mnt/d/MXDiscovery
        """
        wsl_base = "/mnt/d/MXDiscovery"
        wsl_dft_dir = f"{wsl_base}/data/results/dft"
        wsl_pseudo = f"{wsl_base}/data/pseudopotentials"

        lines = [
            "#!/bin/bash",
            "# =================================================================",
            "#  MXDiscovery Stage 6: DFT Validation with Quantum ESPRESSO",
            "# =================================================================",
            "#  Auto-generated master script for WSL2 execution.",
            "#",
            "#  USAGE:",
            "#    From WSL2 terminal:",
            f"#    cd {wsl_dft_dir}",
            "#    bash run_all_dft.sh 2>&1 | tee dft_log.txt",
            "#",
            "#  REQUIREMENTS:",
            "#    - Quantum ESPRESSO installed (sudo apt install quantum-espresso)",
            "#    - Pseudopotentials downloaded to data/pseudopotentials/",
            "#    - Sufficient disk space (~500MB per candidate for tmp files)",
            "# =================================================================",
            "",
            f"NPROCS={nprocs}",
            f"PSEUDO_DIR={wsl_pseudo}",
            "",
            "echo '========================================'",
            "echo '  MXDiscovery DFT Validation Pipeline'",
            "echo '========================================'",
            "echo \"Started: $(date)\"",
            "echo \"Using $NPROCS MPI processes\"",
            "echo ''",
            "",
        ]

        for i, name in enumerate(names, 1):
            calc_dir = f"{wsl_dft_dir}/{name}"
            lines.extend([
                f"# --- Candidate {i}/{len(names)}: {name} ---",
                f"echo '--- [{i}/{len(names)}] {name} ---'",
                f"cd {calc_dir}",
                "mkdir -p tmp",
                "",
                f"# Step 1: SCF (self-consistent field)",
                f"echo \"  SCF starting: $(date)\"",
                f"mpirun -np $NPROCS pw.x < {name}.scf.in > {name}.scf.out 2>&1",
                f"if grep -q 'convergence has been achieved' {name}.scf.out; then",
                f"  echo '  SCF: CONVERGED'",
                f"else",
                f"  echo '  SCF: FAILED - check {name}.scf.out'",
                f"  echo 'Skipping remaining calculations for {name}'",
                # Don't exit, continue to next candidate
                f"  cd {wsl_dft_dir}",
                f"  # skip to next candidate",
                f"  echo ''",
                f"  continue 2>/dev/null || true",
                f"fi",
                "",
                f"# Step 2: NSCF (dense k-grid for DOS)",
                f"echo \"  NSCF starting: $(date)\"",
                f"mpirun -np $NPROCS pw.x < {name}.nscf.in > {name}.nscf.out 2>&1",
                f"echo '  NSCF: done'",
                "",
                f"# Step 3: DOS",
                f"echo \"  DOS starting: $(date)\"",
                f"dos.x < {name}.dos.in > {name}.dos.out 2>&1",
                f"echo '  DOS: done'",
                "",
                f"# Step 4: Bands (k-path)",
                f"echo \"  Bands starting: $(date)\"",
                f"mpirun -np $NPROCS pw.x < {name}.bands.in > {name}.bands.out 2>&1",
                f"echo '  Bands pw.x: done'",
                "",
                f"# Step 5: Bands post-processing",
                f"bands.x < {name}.bands_pp.in > {name}.bands_pp.out 2>&1",
                f"echo '  Bands post-process: done'",
                "",
                f"echo \"  {name} COMPLETE: $(date)\"",
                f"echo ''",
                "",
            ])

        lines.extend([
            "echo '========================================'",
            "echo '  All DFT calculations complete!'",
            "echo \"Finished: $(date)\"",
            "echo '========================================'",
        ])

        master_script = self.output_dir / "run_all_dft.sh"
        with open(master_script, "w", newline="\n") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Generated master DFT script: {master_script}")
        return master_script

    def parse_dos_output(self, dos_file: str | Path) -> dict:
        """
        Parse dos.x output file for bandgap and DOS features.

        DOS FILE FORMAT (space-separated):
            # E (eV)   dos(E)   pdos(E)
            -10.000    0.0000   0.0000
            ...

        BANDGAP DETECTION:
            1. Find the Fermi energy region
            2. Look for energy range where DOS ~ 0 around Fermi level
            3. If DOS never drops to ~0 near E_F -> metallic (gap = 0)
            4. Otherwise, gap = E_CBM - E_VBM

        RETURNS:
            {
                "bandgap_ev": float or 0.0,
                "is_metallic": bool,
                "dos_at_fermi": float,
                "vbm_ev": float,  # valence band maximum
                "cbm_ev": float,  # conduction band minimum
            }
        """
        dos_file = Path(dos_file)
        if not dos_file.exists():
            return {"error": f"File not found: {dos_file}"}

        energies = []
        dos_values = []

        with open(dos_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        energies.append(float(parts[0]))
                        dos_values.append(float(parts[1]))
                    except ValueError:
                        continue

        if not energies:
            return {"error": "No DOS data found"}

        import numpy as np
        energies = np.array(energies)
        dos_values = np.array(dos_values)

        # Find DOS at Fermi level (E=0 in QE dos output, relative to E_F)
        fermi_idx = np.argmin(np.abs(energies))
        dos_at_fermi = dos_values[fermi_idx]

        # Detect bandgap: find region near E_F where DOS < threshold
        threshold = 0.01 * np.max(dos_values)  # 1% of max DOS

        # Look for gap around Fermi level
        near_fermi = np.abs(energies) < 5.0  # within 5 eV of Fermi
        low_dos = dos_values < threshold

        # Find VBM (highest energy below E_F with significant DOS)
        below_fermi = (energies < 0) & near_fermi
        if np.any(below_fermi & ~low_dos):
            vbm_candidates = energies[below_fermi & ~low_dos]
            vbm = np.max(vbm_candidates)
        else:
            vbm = 0.0

        # Find CBM (lowest energy above E_F with significant DOS)
        above_fermi = (energies > 0) & near_fermi
        if np.any(above_fermi & ~low_dos):
            cbm_candidates = energies[above_fermi & ~low_dos]
            cbm = np.min(cbm_candidates)
        else:
            cbm = 0.0

        bandgap = max(0.0, cbm - vbm)
        is_metallic = dos_at_fermi > threshold

        return {
            "bandgap_ev": bandgap if not is_metallic else 0.0,
            "is_metallic": bool(is_metallic),
            "dos_at_fermi": float(dos_at_fermi),
            "vbm_ev": float(vbm),
            "cbm_ev": float(cbm),
        }
