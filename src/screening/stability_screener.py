"""
stability_screener.py - Screen MXene structures for thermodynamic stability
                        using ML interatomic potentials (CHGNet / MACE)

ALGORITHM (Overview):
    1. Load candidate structure (from structure_generator.py)
    2. Relax the structure using ML potential (find lowest energy configuration)
    3. Calculate formation energy (is this structure thermodynamically stable?)
    4. Compare to known stable phases (energy above convex hull)
    5. Filter: keep only structures with formation energy below threshold

WHAT IS STRUCTURE RELAXATION?
    Atoms in a generated structure are placed at idealized positions.
    In reality, they shift slightly to minimize the total energy.
    Relaxation = iteratively move atoms along force directions until
    forces on all atoms are below a threshold.

    Think of it like dropping a ball on a hilly landscape - it rolls
    to the nearest valley (local energy minimum).

    CHGNet/MACE predict the energy AND forces for any atomic configuration,
    so we can relax without expensive DFT calculations.

WHAT IS FORMATION ENERGY?
    E_formation = E_compound - sum(E_elements)
    Negative = compound is stable (lower energy than separated elements)
    Positive = compound is unstable (wants to decompose)

    We normalize per atom: E_f / N_atoms (eV/atom)
    Typical stable materials: -0.5 to -3.0 eV/atom
    Threshold for "possibly stable": < 0.2 eV/atom above hull

WHAT IS CHGNet?
    Crystal Hamiltonian Graph Neural Network (2023, Berkeley)
    - Universal ML potential trained on Materials Project DFT data
    - Input: crystal structure (atom types + positions + cell)
    - Output: energy, forces on each atom, stress tensor
    - Accuracy: ~30 meV/atom MAE vs DFT
    - Speed: ~1000x faster than DFT
    - Size: ~400 MB model, fits in GPU memory easily
    - Perfect for screening hundreds of candidates quickly

WHAT IS MACE?
    Multi-Atomic Cluster Expansion (2022, Cambridge)
    - Another universal ML potential, slightly more accurate
    - Uses equivariant message passing
    - MACE-MP-0: foundation model trained on Materials Project
    - Accuracy: ~20 meV/atom MAE vs DFT
    - Slower than CHGNet but more accurate

TECHNIQUES:
    - ASE optimization: BFGS/FIRE algorithms for structure relaxation
    - GPU acceleration: both CHGNet and MACE support CUDA
    - Batch processing: relax multiple structures sequentially
    - Checkpointing: save relaxed structures after each calculation
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    from ase import Atoms
    from ase.io import write as ase_write, read as ase_read
    from ase.optimize import BFGS, FIRE
except ImportError:
    logger.warning("ASE not installed.")
    Atoms = None


class StabilityScreener:
    """
    Screens MXene structures using ML interatomic potentials.

    HOW IT WORKS:
        1. Load ML model (CHGNet or MACE) into GPU memory (done once)
        2. For each candidate structure:
           a. Attach the ML calculator to the ASE Atoms object
           b. Run structural relaxation (BFGS optimizer)
           c. Read final energy, forces, stress
           d. Calculate formation energy
           e. Determine if stable (below energy threshold)
        3. Save relaxed structures and results

    USAGE:
        screener = StabilityScreener(model="chgnet")
        result = screener.screen_one(atoms, name="Ti3C2O2")
        # result = {"energy": -45.2, "energy_per_atom": -5.65, "is_stable": True, ...}
    """

    # Reference energies per atom for elemental ground states (eV/atom)
    # From Materials Project, needed to compute formation energies
    # These are approximate - for publication, use exact MP values
    REFERENCE_ENERGIES = {
        "Ti": -7.895, "V": -8.941, "Cr": -9.508, "Mn": -8.998,
        "Zr": -8.547, "Nb": -10.094, "Mo": -10.847, "Hf": -9.955,
        "Ta": -11.853, "W": -12.960, "Sc": -6.333,
        "C": -9.227, "N": -8.336,
        "O": -4.948, "F": -1.912, "Cl": -1.816, "S": -4.125,
        "Se": -3.495, "Te": -3.143, "Br": -1.640,
        "H": -3.393,
    }

    def __init__(
        self,
        model: str = "chgnet",
        device: str = "cuda",
        energy_cutoff: float = 0.2,
        force_tol: float = 0.05,
        max_steps: int = 500,
        output_dir: str | Path = "data/structures/relaxed",
    ):
        """
        PARAMS:
            model: "chgnet" or "mace"
            device: "cuda" for GPU, "cpu" for CPU
            energy_cutoff: eV/atom above hull to consider "possibly stable"
            force_tol: eV/Å - relaxation converges when max force < this
            max_steps: maximum relaxation steps before giving up
        """
        self.model_name = model
        self.device = device
        self.energy_cutoff = energy_cutoff
        self.force_tol = force_tol
        self.max_steps = max_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.calculator = self._load_model()

    def _load_model(self):
        """
        Load the ML potential model.

        CHGNet LOADING:
            1. Import CHGNetCalculator from chgnet.model
            2. It auto-downloads the pretrained model (~400MB) on first use
            3. Model is loaded to GPU if available
            4. Returns an ASE-compatible Calculator object

        MACE LOADING:
            1. Import MACECalculator from mace.calculators
            2. Load MACE-MP-0 foundation model
            3. Similar ASE Calculator interface
        """
        if self.model_name == "chgnet":
            try:
                from chgnet.model.dynamics import CHGNetCalculator
                calc = CHGNetCalculator(use_device=self.device)
                logger.info(f"CHGNet loaded on {self.device}")
                return calc
            except ImportError:
                logger.error("CHGNet not installed. Run: pip install chgnet")
                raise
            except Exception as e:
                logger.warning(f"GPU load failed, falling back to CPU: {e}")
                from chgnet.model.dynamics import CHGNetCalculator
                calc = CHGNetCalculator(use_device="cpu")
                return calc

        elif self.model_name == "mace":
            try:
                from mace.calculators import MACECalculator
                calc = MACECalculator(
                    model_paths="medium",  # MACE-MP-0 medium
                    device=self.device,
                    default_dtype="float32",
                )
                logger.info(f"MACE loaded on {self.device}")
                return calc
            except ImportError:
                logger.error("MACE not installed. Run: pip install mace-torch")
                raise

        else:
            raise ValueError(f"Unknown model: {self.model_name}. Use 'chgnet' or 'mace'.")

    def _compute_formation_energy(self, atoms: "Atoms", total_energy: float) -> float:
        """
        Compute formation energy per atom.

        FORMULA:
            E_f = (E_total - sum(n_i * E_ref_i)) / N_total

        WHERE:
            E_total = total energy of the compound (from ML potential)
            n_i = number of atoms of element i in the structure
            E_ref_i = reference energy per atom for element i (bulk ground state)
            N_total = total number of atoms

        EXAMPLE for Ti3C2O2 (8 atoms):
            E_f = (E_total - 3*E_Ti - 2*E_C - 2*E_O) / 8
        """
        from collections import Counter
        composition = Counter(atoms.get_chemical_symbols())
        n_total = len(atoms)

        ref_energy = 0.0
        for element, count in composition.items():
            if element in self.REFERENCE_ENERGIES:
                ref_energy += count * self.REFERENCE_ENERGIES[element]
            else:
                logger.warning(f"No reference energy for {element}, using 0")

        formation_energy = (total_energy - ref_energy) / n_total
        return formation_energy

    def relax(self, atoms: "Atoms", name: str = "structure") -> dict:
        """
        Relax a structure and compute properties.

        ALGORITHM (step by step):
            1. Attach ML calculator to atoms object
            2. Create BFGS optimizer (quasi-Newton method)
               - BFGS = Broyden-Fletcher-Goldfarb-Shanno
               - Builds approximate Hessian matrix from gradient history
               - Efficient for smooth energy landscapes
            3. Run optimizer until:
               - Max force on any atom < force_tol, OR
               - Max steps reached (not converged)
            4. Read final energy, forces, stress from relaxed structure
            5. Compute formation energy
            6. Determine stability: formation_energy < energy_cutoff

        RETURNS: dict with all computed properties
        """
        if Atoms is None:
            raise ImportError("ASE required")

        atoms = atoms.copy()  # Don't modify original
        atoms.calc = self.calculator

        # Get initial energy
        try:
            initial_energy = atoms.get_potential_energy()
        except Exception as e:
            logger.error(f"Initial energy calculation failed for {name}: {e}")
            return {"name": name, "error": str(e), "is_stable": False}

        # Relax
        logfile = str(self.output_dir / f"{name}_relax.log")
        optimizer = BFGS(atoms, logfile=logfile)

        try:
            converged = optimizer.run(fmax=self.force_tol, steps=self.max_steps)
        except Exception as e:
            logger.error(f"Relaxation failed for {name}: {e}")
            return {"name": name, "error": str(e), "is_stable": False}

        # Read results
        final_energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))

        # Formation energy
        formation_energy = self._compute_formation_energy(atoms, final_energy)
        is_stable = formation_energy < self.energy_cutoff

        # Save relaxed structure
        relaxed_file = self.output_dir / f"{name}_relaxed.cif"
        ase_write(str(relaxed_file), atoms, format="cif")

        result = {
            "name": name,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_per_atom": final_energy / len(atoms),
            "formation_energy": formation_energy,
            "max_force": max_force,
            "converged": max_force < self.force_tol,
            "n_steps": optimizer.nsteps,
            "n_atoms": len(atoms),
            "composition": dict(zip(*np.unique(atoms.get_chemical_symbols(), return_counts=True))),
            "is_stable": is_stable,
            "relaxed_structure": str(relaxed_file),
            "model": self.model_name,
        }

        stability = "STABLE" if is_stable else "UNSTABLE"
        logger.info(
            f"  {name}: E_f={formation_energy:.4f} eV/atom [{stability}] "
            f"({'converged' if result['converged'] else 'not converged'}, {optimizer.nsteps} steps)"
        )
        return result

    def screen_batch(self, structures: list[dict]) -> list[dict]:
        """
        Screen a batch of structures for stability.

        INPUT: list of dicts with 'atoms' and 'name' keys
               (output of structure_generator.generate_batch())

        ALGORITHM:
            1. Sort by expected complexity (fewer atoms first = faster)
            2. Relax each structure sequentially
            3. Collect results, mark stable/unstable
            4. Sort results by formation energy (most stable first)

        NOTE: We process sequentially because:
            - GPU can only handle one calculation at a time
            - Each relaxation takes 10-60 seconds
            - For 50 candidates, expect ~30 minutes total
        """
        results = []
        n_total = len(structures)

        # Sort by atom count (faster calculations first for quick feedback)
        structures.sort(key=lambda s: s.get("n_atoms", 999))

        for i, struct in enumerate(structures):
            if struct.get("atoms") is None:
                logger.warning(f"Skipping {struct.get('name', '?')} - no atoms object")
                continue

            name = struct.get("name", f"candidate_{i}")
            logger.info(f"Screening {i+1}/{n_total}: {name}")

            result = self.relax(struct["atoms"], name=name)
            result.update({
                "mxene_formula": struct.get("mxene_formula", ""),
                "termination": struct.get("termination", ""),
                "composite_partner": struct.get("composite_partner", ""),
            })
            results.append(result)

        # Sort by formation energy (most stable first)
        results.sort(key=lambda r: r.get("formation_energy", 999))

        # Summary
        n_stable = sum(1 for r in results if r.get("is_stable"))
        logger.info(f"\nScreening complete: {n_stable}/{len(results)} stable candidates")
        return results
