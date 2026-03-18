"""
structure_generator.py - Generate MXene crystal structures programmatically

ALGORITHM:
    MXenes have a layered hexagonal structure derived from MAX phases.
    The parent MAX phase has formula M_{n+1}AX_n where:
    - M = early transition metal
    - A = group IIIA/IVA element (removed during etching)
    - X = C or N

    After etching the A layer, we get M_{n+1}X_n with surface terminations (Tx).
    The structure is:
    - Hexagonal unit cell (space group P6_3/mmc for M2X, P-3m1 for terminated)
    - ABC stacking of M-X layers
    - Surface terminations sit in hollow sites on the outer M layers

    STEP-BY-STEP STRUCTURE GENERATION:

    1. BUILD BASE M_{n+1}X_n SLAB:
       a. Start with hexagonal lattice parameter 'a' (from known data or estimate)
       b. Place M atoms at hexagonal positions in each M layer
       c. Place X atoms in octahedral holes between M layers
       d. Stack layers with appropriate c-spacing

    2. ADD SURFACE TERMINATIONS:
       a. Identify hollow sites on top and bottom M surfaces
       b. Place termination atoms/groups (O, OH, F, etc.) at these sites
       c. Three possible sites: fcc, hcp, atop -> most stable is typically fcc
       d. We default to fcc hollow sites (can be relaxed later with DFT/ML)

    3. ADD VACUUM:
       a. Add vacuum layer along c-axis (typically 20 Å)
       b. This prevents interaction between periodic images

    4. FOR COMPOSITES:
       a. Build the MXene slab
       b. Build the partner material slab/molecule
       c. Stack them with appropriate interlayer distance
       d. This is for interface calculations

DATA STRUCTURES:
    - ASE Atoms object: the fundamental structure representation
      Contains: atomic positions, cell vectors, species, periodic boundary conditions
    - CIF / POSCAR files: standard crystallographic interchange formats

TECHNIQUES:
    - Lattice parameter estimation from Shannon ionic radii
    - Wyckoff position assignment for high-symmetry structures
    - Automatic vacuum padding for 2D slab calculations
    - Structure validation: check bond lengths, no overlapping atoms

KNOWN LATTICE PARAMETERS (Å) from DFT literature:
    Ti2CTx:   a=3.04, Ti-C=2.16
    Ti3C2Tx:  a=3.07, Ti-C=2.16, c_internal=2.33
    V2CTx:    a=2.91, V-C=2.06
    Nb2CTx:   a=3.15, Nb-C=2.23
    Mo2CTx:   a=2.86, Mo-C=2.12
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    from ase import Atoms
    from ase.io import write as ase_write, read as ase_read
    from ase.build import surface, make_supercell
    from ase.constraints import FixAtoms
except ImportError:
    logger.warning("ASE not installed. Structure generation disabled.")
    Atoms = None


# ---------------------------------------------------------------------------
# Known lattice parameters from DFT calculations in literature
# Format: {formula: {"a": Å, "m_x_bond": Å, "m_layer_sep": Å}}
# ---------------------------------------------------------------------------
LATTICE_PARAMS = {
    "Ti2C":   {"a": 3.04, "m_x_bond": 2.16, "m_layer_sep": 2.33},
    "Ti3C2":  {"a": 3.07, "m_x_bond": 2.16, "m_layer_sep": 2.33},
    "Ti4C3":  {"a": 3.08, "m_x_bond": 2.17, "m_layer_sep": 2.34},
    "V2C":    {"a": 2.91, "m_x_bond": 2.06, "m_layer_sep": 2.22},
    "V2N":    {"a": 2.97, "m_x_bond": 2.10, "m_layer_sep": 2.25},
    "Nb2C":   {"a": 3.15, "m_x_bond": 2.23, "m_layer_sep": 2.40},
    "Mo2C":   {"a": 2.86, "m_x_bond": 2.12, "m_layer_sep": 2.28},
    "Cr2C":   {"a": 2.84, "m_x_bond": 2.08, "m_layer_sep": 2.20},
    "Zr2C":   {"a": 3.29, "m_x_bond": 2.34, "m_layer_sep": 2.50},
    "Hf2C":   {"a": 3.23, "m_x_bond": 2.30, "m_layer_sep": 2.47},
    "Ta2C":   {"a": 3.10, "m_x_bond": 2.20, "m_layer_sep": 2.36},
    "W2C":    {"a": 2.85, "m_x_bond": 2.11, "m_layer_sep": 2.27},
    "Sc2C":   {"a": 3.36, "m_x_bond": 2.40, "m_layer_sep": 2.55},
}

# Termination bond lengths (Å) from DFT
# Distance from outer M atom to termination atom
TERMINATION_DISTANCES = {
    "O":  1.10,
    "OH": 1.15,
    "F":  1.20,
    "Cl": 1.60,
    "S":  1.50,
    "Se": 1.65,
    "Te": 1.85,
    "Br": 1.75,
}


class MXeneStructureGenerator:
    """
    Generates MXene crystal structures for computational screening.

    HOW IT WORKS (detailed):

    The MXene unit cell is hexagonal with atoms at specific Wyckoff positions.
    For M2XTx (simplest case, n=1):

        Layer structure (side view along a-axis):
            Tx   ← termination (top)
            M    ← outer metal layer (top)
            X    ← carbon/nitrogen
            M    ← outer metal layer (bottom)
            Tx   ← termination (bottom)

        Top view: hexagonal arrangement
            M atoms at (0,0) and (1/3, 2/3)
            X atoms at (2/3, 1/3) - octahedral holes
            T atoms at (1/3, 2/3) or (0,0) - fcc hollow sites

    For M3X2Tx (n=2):
            Tx
            M    ← outer
            X
            M    ← inner
            X
            M    ← outer
            Tx

    The generator:
    1. Looks up lattice parameter 'a' from the table above
    2. Places atoms layer by layer along the c-axis
    3. Adds vacuum padding
    4. Writes to CIF/POSCAR format for use in CHGNet/DFT
    """

    def __init__(self, output_dir: str | Path = "data/structures", vacuum: float = 20.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vacuum = vacuum  # Å of vacuum for slab calculations

    def _estimate_lattice_params(self, m_element: str, x_element: str) -> dict:
        """
        Get or estimate lattice parameters for a given MXene.

        ALGORITHM:
            1. Check exact match in LATTICE_PARAMS table
            2. If not found, estimate from M2C entry (most data available)
            3. If M element not in table at all, use Vegard's law interpolation
               based on atomic radius

        Vegard's law: a_alloy ≈ x*a_1 + (1-x)*a_2
        For single M elements, we interpolate from nearest known values.
        """
        base = f"{m_element}2{x_element}"
        if base in LATTICE_PARAMS:
            return LATTICE_PARAMS[base]

        # Fallback: estimate from atomic radius relative to Ti
        # Shannon ionic radii for M^3+ in octahedral coordination (Å)
        radii = {
            "Ti": 0.670, "V": 0.640, "Cr": 0.615, "Mn": 0.645,
            "Zr": 0.720, "Nb": 0.680, "Mo": 0.650, "Hf": 0.710,
            "Ta": 0.680, "W": 0.620, "Sc": 0.745,
        }
        r_m = radii.get(m_element, 0.670)  # default to Ti
        r_ti = radii["Ti"]
        ratio = r_m / r_ti

        ti_params = LATTICE_PARAMS["Ti2C"]
        return {
            "a": ti_params["a"] * ratio,
            "m_x_bond": ti_params["m_x_bond"] * ratio,
            "m_layer_sep": ti_params["m_layer_sep"] * ratio,
        }

    def generate_m2x(
        self,
        m_element: str,
        x_element: str = "C",
        termination: Optional[str] = None,
    ) -> "Atoms":
        """
        Generate M2XTx structure (n=1, simplest MXene).

        ATOM POSITIONS (fractional coordinates):
            M1: (0, 0, z_m_top)
            M2: (1/3, 2/3, z_m_bot)
            X:  (2/3, 1/3, z_x)       - center
            T1: (2/3, 1/3, z_t_top)   - top termination (fcc hollow)
            T2: (0, 0, z_t_bot)       - bottom termination (fcc hollow)

        The z-coordinates are computed from bond lengths.
        """
        if Atoms is None:
            raise ImportError("ASE is required. Install with: pip install ase")

        params = self._estimate_lattice_params(m_element, x_element)
        a = params["a"]
        bond = params["m_x_bond"]
        sep = params["m_layer_sep"]

        # Calculate z positions (Å from bottom of slab)
        z_center = self.vacuum / 2  # center of vacuum
        z_x = z_center
        z_m_bot = z_center - sep / 2
        z_m_top = z_center + sep / 2

        # Hexagonal cell
        cell = [
            [a, 0, 0],
            [a * np.cos(np.pi * 2 / 3), a * np.sin(np.pi * 2 / 3), 0],
            [0, 0, self.vacuum + sep + 4],  # total c with vacuum
        ]

        # Atom positions (Cartesian)
        positions = []
        symbols = []

        # M atoms (2 per unit cell)
        # M1 at fractional (0, 0)
        positions.append([0, 0, z_m_bot])
        symbols.append(m_element)
        # M2 at fractional (1/3, 2/3)
        pos_m2 = np.array(cell[0]) * 1 / 3 + np.array(cell[1]) * 2 / 3
        positions.append([pos_m2[0], pos_m2[1], z_m_top])
        symbols.append(m_element)

        # X atom at fractional (2/3, 1/3)
        pos_x = np.array(cell[0]) * 2 / 3 + np.array(cell[1]) * 1 / 3
        positions.append([pos_x[0], pos_x[1], z_x])
        symbols.append(x_element)

        # Termination atoms
        if termination and termination != "bare":
            t_dist = TERMINATION_DISTANCES.get(termination, 1.2)
            t_element = termination.replace("OH", "O")  # simplified

            # Top termination at fcc hollow (2/3, 1/3)
            positions.append([pos_x[0], pos_x[1], z_m_top + t_dist])
            symbols.append(t_element)

            # Bottom termination at fcc hollow (0, 0)
            positions.append([0, 0, z_m_bot - t_dist])
            symbols.append(t_element)

            if termination == "OH":
                # Add H atoms for hydroxyl termination
                positions.append([pos_x[0], pos_x[1], z_m_top + t_dist + 0.97])
                symbols.append("H")
                positions.append([0, 0, z_m_bot - t_dist - 0.97])
                symbols.append("H")

        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=[True, True, False],  # periodic in x,y; vacuum in z
        )

        return atoms

    def generate_m3x2(
        self,
        m_element: str,
        x_element: str = "C",
        termination: Optional[str] = None,
    ) -> "Atoms":
        """
        Generate M3X2Tx structure (n=2, most common MXene e.g. Ti3C2Tx).

        Layer stacking:
            T_top -> M_top -> X_top -> M_mid -> X_bot -> M_bot -> T_bot

        3 metal layers, 2 carbon layers, 2 termination sites.
        """
        if Atoms is None:
            raise ImportError("ASE is required.")

        params = self._estimate_lattice_params(m_element, x_element)
        a = params["a"]
        sep = params["m_layer_sep"]

        z_center = self.vacuum / 2
        # 3 M layers, 2 X layers
        z_m_mid = z_center
        z_x_bot = z_center - sep / 2
        z_x_top = z_center + sep / 2
        z_m_bot = z_center - sep
        z_m_top = z_center + sep

        cell = [
            [a, 0, 0],
            [a * np.cos(np.pi * 2 / 3), a * np.sin(np.pi * 2 / 3), 0],
            [0, 0, self.vacuum + 2 * sep + 6],
        ]

        positions = []
        symbols = []

        # Fractional position helpers
        pos_A = np.array([0, 0, 0])  # (0, 0)
        pos_B = np.array(cell[0]) * 1 / 3 + np.array(cell[1]) * 2 / 3  # (1/3, 2/3)
        pos_C = np.array(cell[0]) * 2 / 3 + np.array(cell[1]) * 1 / 3  # (2/3, 1/3)

        # M layers (ABC stacking)
        positions.append([pos_A[0], pos_A[1], z_m_bot]); symbols.append(m_element)
        positions.append([pos_B[0], pos_B[1], z_m_mid]); symbols.append(m_element)
        positions.append([pos_C[0], pos_C[1], z_m_top]); symbols.append(m_element)

        # X layers
        positions.append([pos_C[0], pos_C[1], z_x_bot]); symbols.append(x_element)
        positions.append([pos_A[0], pos_A[1], z_x_top]); symbols.append(x_element)

        # Terminations
        if termination and termination != "bare":
            t_dist = TERMINATION_DISTANCES.get(termination, 1.2)
            t_element = termination.replace("OH", "O")

            positions.append([pos_B[0], pos_B[1], z_m_top + t_dist])
            symbols.append(t_element)
            positions.append([pos_C[0], pos_C[1], z_m_bot - t_dist])
            symbols.append(t_element)

            if termination == "OH":
                positions.append([pos_B[0], pos_B[1], z_m_top + t_dist + 0.97])
                symbols.append("H")
                positions.append([pos_C[0], pos_C[1], z_m_bot - t_dist - 0.97])
                symbols.append("H")

        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=[True, True, False],
        )
        return atoms

    def generate(
        self,
        m_element: str,
        x_element: str = "C",
        n: int = 2,
        termination: Optional[str] = "O",
        supercell: Optional[tuple] = None,
    ) -> "Atoms":
        """
        Generate MXene structure with given parameters.

        PARAMS:
            m_element: transition metal (e.g., "Ti", "V", "Mo")
            x_element: "C" or "N"
            n: MXene order (1 -> M2X, 2 -> M3X2, 3 -> M4X3)
            termination: surface termination ("O", "OH", "F", etc.) or None
            supercell: optional (nx, ny, nz) for supercell expansion

        RETURNS: ASE Atoms object
        """
        if n == 1:
            atoms = self.generate_m2x(m_element, x_element, termination)
        elif n == 2:
            atoms = self.generate_m3x2(m_element, x_element, termination)
        else:
            # n=3 (M4X3) - extend the pattern with 4 M layers, 3 X layers
            logger.warning(f"M4X3 generation uses simplified model")
            atoms = self._generate_m4x3(m_element, x_element, termination)

        if supercell:
            atoms = make_supercell(atoms, np.diag(supercell))

        return atoms

    def _generate_m4x3(self, m_element, x_element, termination):
        """Generate M4X3Tx (n=3). Same logic as M3X2 but with one more M-X layer."""
        params = self._estimate_lattice_params(m_element, x_element)
        a = params["a"]
        sep = params["m_layer_sep"]

        z_center = self.vacuum / 2
        z_positions_m = [z_center - 1.5 * sep, z_center - 0.5 * sep,
                         z_center + 0.5 * sep, z_center + 1.5 * sep]
        z_positions_x = [z_center - sep, z_center, z_center + sep]

        cell = [
            [a, 0, 0],
            [a * np.cos(np.pi * 2 / 3), a * np.sin(np.pi * 2 / 3), 0],
            [0, 0, self.vacuum + 3 * sep + 8],
        ]

        pos_A = np.array([0, 0, 0])
        pos_B = np.array(cell[0]) * 1 / 3 + np.array(cell[1]) * 2 / 3
        pos_C = np.array(cell[0]) * 2 / 3 + np.array(cell[1]) * 1 / 3

        hex_positions = [pos_A, pos_B, pos_C, pos_A]  # ABCA stacking for M
        hex_positions_x = [pos_C, pos_A, pos_B]  # octahedral holes

        positions = []
        symbols = []

        for z, hp in zip(z_positions_m, hex_positions):
            positions.append([hp[0], hp[1], z])
            symbols.append(m_element)

        for z, hp in zip(z_positions_x, hex_positions_x):
            positions.append([hp[0], hp[1], z])
            symbols.append(x_element)

        if termination and termination != "bare":
            t_dist = TERMINATION_DISTANCES.get(termination, 1.2)
            t_element = termination.replace("OH", "O")
            positions.append([pos_B[0], pos_B[1], z_positions_m[-1] + t_dist])
            symbols.append(t_element)
            positions.append([pos_C[0], pos_C[1], z_positions_m[0] - t_dist])
            symbols.append(t_element)

        return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=[True, True, False])

    def save_structure(self, atoms: "Atoms", name: str, fmt: str = "vasp") -> Path:
        """
        Save structure to file.

        FORMATS:
            "vasp" -> POSCAR (for VASP/CHGNet)
            "cif"  -> CIF (universal crystallographic format)
            "xyz"  -> XYZ (simple, human-readable)
        """
        ext_map = {"vasp": "POSCAR", "cif": ".cif", "xyz": ".xyz"}
        if fmt == "vasp":
            filename = self.output_dir / f"{name}_POSCAR"
        else:
            filename = self.output_dir / f"{name}{ext_map[fmt]}"

        ase_write(str(filename), atoms, format=fmt)
        logger.info(f"Saved structure: {filename}")
        return filename

    def generate_batch(
        self,
        candidates: list[dict],
        fmt: str = "vasp",
    ) -> list[dict]:
        """
        Generate structures for a batch of candidates from gap analysis.

        INPUT: list of dicts with keys:
            m_elements, x_element, stoichiometry, termination, mxene_formula

        RETURNS: list of dicts with added 'structure_file' and 'atoms' keys
        """
        results = []
        for cand in candidates:
            m_el = cand["m_elements"][0] if isinstance(cand["m_elements"], list) else cand["m_elements"]
            x_el = cand.get("x_element", "C")
            stoich = cand.get("stoichiometry", "M3X2")
            term = cand.get("termination", "O")
            partner = cand.get("composite_partner", "")

            n_map = {"M2X": 1, "M3X2": 2, "M4X3": 3}
            n = n_map.get(stoich, 2)

            try:
                atoms = self.generate(m_el, x_el, n=n, termination=term)
                name = f"{cand.get('mxene_formula', f'{m_el}{n+1}{x_el}{n}')}_{term}_{partner}"
                name = name.replace(":", "_").replace("/", "_")
                filepath = self.save_structure(atoms, name, fmt=fmt)
                results.append({
                    **cand,
                    "structure_file": str(filepath),
                    "atoms": atoms,
                    "n_atoms": len(atoms),
                })
            except Exception as e:
                logger.error(f"Failed to generate {cand}: {e}")
                results.append({**cand, "structure_file": None, "error": str(e)})

        logger.info(f"Generated {sum(1 for r in results if r.get('structure_file'))} / {len(candidates)} structures")
        return results
