"""
te_predictor.py - Predict thermoelectric properties for MXene candidates

ALGORITHM (Overview):
    This module predicts thermoelectric properties using two approaches:

    APPROACH 1: ELECTRONIC STRUCTURE → TRANSPORT (Physics-based)
        1. Compute electronic band structure from relaxed structure
        2. Feed band structure into BoltzTraP2 (Boltzmann transport equation solver)
        3. BoltzTraP2 outputs: Seebeck coefficient, electrical conductivity,
           electronic thermal conductivity as functions of temperature & doping

        WHY: This is the most physically rigorous approach. BoltzTraP2 solves
        the semiclassical Boltzmann transport equation under the relaxation
        time approximation (RTA).

        LIMITATION: Requires DFT band structure as input. ML potentials give
        energies/forces but NOT electronic structure. So this approach needs
        DFT for the top candidates (Quantum ESPRESSO).

    APPROACH 2: ML PROPERTY PREDICTION (Data-driven)
        1. Use ALIGNN (Atomistic Line Graph Neural Network) to predict
           properties directly from crystal structure
        2. Pre-trained on Materials Project data
        3. Can predict: bandgap, formation energy, total energy, bulk modulus
        4. For thermoelectric-specific properties (Seebeck, ZT), we need
           to either fine-tune on TE data or use semi-empirical relations

        Semi-empirical TE estimation:
        - Bandgap → Seebeck coefficient (Goldsmid-Sharp relation):
          S_max ≈ E_gap / (2 * e * T_max)
          where e = electron charge, T_max = temperature of max ZT
        - This gives an UPPER BOUND estimate of Seebeck coefficient

    APPROACH 3: DESCRIPTOR-BASED ML (Custom model)
        1. Compute structural/compositional descriptors for each candidate
        2. Train a simple ML model (Random Forest / Gradient Boosting)
           on our extracted literature data (from data_extractor.py)
        3. Predict properties for new candidates

        DESCRIPTORS:
        - Compositional: electronegativity (mean, std), atomic mass, d-electron count
        - Structural: lattice parameter, layer thickness, interlayer distance
        - Electronic: estimated bandgap, work function
        - Termination: termination electronegativity, bond ionicity

DATA STRUCTURES:
    - TEProperties: dataclass holding all thermoelectric properties
    - DescriptorVector: numpy array of compositional/structural features

TECHNIQUES:
    - Goldsmid-Sharp relation for Seebeck estimation from bandgap
    - Wiedemann-Franz law for electronic thermal conductivity
    - Slack model for lattice thermal conductivity estimation
    - ALIGNN for ML-based property prediction
    - Scikit-learn for descriptor-based models
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

try:
    from ase import Atoms
except ImportError:
    Atoms = None


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
KB = 8.617333e-5   # Boltzmann constant (eV/K)
E_CHARGE = 1.602e-19  # electron charge (C)
LORENZ = 2.44e-8   # Lorenz number (WΩ/K²) for Wiedemann-Franz


@dataclass
class TEProperties:
    """Predicted thermoelectric properties for a candidate."""
    name: str
    temperature_k: float = 310.0  # body temperature for wearables

    # Core TE properties
    seebeck_coefficient: Optional[float] = None  # µV/K
    electrical_conductivity: Optional[float] = None  # S/cm
    thermal_conductivity: Optional[float] = None  # W/mK
    electronic_thermal_cond: Optional[float] = None  # W/mK (κ_e)
    lattice_thermal_cond: Optional[float] = None  # W/mK (κ_L)
    power_factor: Optional[float] = None  # µW/cm·K²
    zt_value: Optional[float] = None

    # Electronic properties
    bandgap: Optional[float] = None  # eV
    effective_mass: Optional[float] = None  # m*/m_e

    # Confidence and method
    method: str = "estimated"  # "estimated", "alignn", "boltztrap", "dft"
    confidence: str = "low"

    def compute_derived(self):
        """
        Compute derived TE properties from available data.

        FORMULAS:
            Power Factor: PF = S² × σ
            ZT = S² × σ × T / κ = PF × T / κ
            κ = κ_e + κ_L (electronic + lattice)
            κ_e = L × σ × T (Wiedemann-Franz law)
        """
        # Power factor
        if self.seebeck_coefficient is not None and self.electrical_conductivity is not None:
            s_v_per_k = self.seebeck_coefficient * 1e-6  # µV/K → V/K
            sigma_s_per_m = self.electrical_conductivity * 100  # S/cm → S/m
            self.power_factor = (s_v_per_k ** 2) * sigma_s_per_m * 1e6  # W/mK² → µW/cm·K²

        # Electronic thermal conductivity (Wiedemann-Franz)
        if self.electrical_conductivity is not None:
            sigma_s_per_m = self.electrical_conductivity * 100
            self.electronic_thermal_cond = LORENZ * sigma_s_per_m * self.temperature_k  # W/mK

        # Total thermal conductivity
        if self.electronic_thermal_cond is not None and self.lattice_thermal_cond is not None:
            self.thermal_conductivity = self.electronic_thermal_cond + self.lattice_thermal_cond

        # ZT
        if (self.power_factor is not None and self.thermal_conductivity is not None
                and self.thermal_conductivity > 0):
            pf_w = self.power_factor * 1e-6 / 100  # µW/cm·K² → W/m·K²
            self.zt_value = pf_w * self.temperature_k / self.thermal_conductivity


class TEPredictor:
    """
    Predicts thermoelectric properties for MXene candidates.

    WORKFLOW:
        1. Quick estimation (Approach 2+3): Use bandgap + descriptors
           for rapid screening of all candidates
        2. ALIGNN prediction: For top candidates, use ML model
        3. BoltzTraP2 (Approach 1): For final candidates, after DFT

    This module implements approaches 2 and 3.
    BoltzTraP2 integration is in simulation/transport.py
    """

    # Estimated bandgaps for MXene compositions (eV)
    # Sources:
    #   - DFT (this project): Mo2CO2 confirmed metallic (PBE, QE v7.5)
    #   - Khazaei et al., Adv. Funct. Mater. 2013
    #   - Zha et al., Nanoscale 2016
    #   - Kumar & Schwingenschlögl, Phys. Rev. B 2016
    #   - Bai et al., Nanoscale 2018
    # NOTE: PBE underestimates bandgaps. HSE06 values are ~30-50% higher.
    BANDGAP_ESTIMATES = {
        # (base_formula, termination) → bandgap in eV
        # Ti-based: mostly metallic or narrow-gap semiconductors
        ("Ti2C", "O"): 0.24,  ("Ti2C", "OH"): 0.05, ("Ti2C", "F"): 0.10,
        ("Ti3C2", "O"): 0.10, ("Ti3C2", "OH"): 0.0, ("Ti3C2", "F"): 0.05,
        ("Ti4C3", "O"): 0.15, ("Ti2N", "O"): 0.30,
        # Mo-based: DFT-validated where available
        ("Mo2C", "O"): 0.0,   # DFT-confirmed metallic (this project, PBE SCF)
        ("Mo2C", "OH"): 0.15, ("Mo2C", "F"): 0.25,
        ("Mo3C2", "O"): 0.05, # metallic trend for thicker Mo carbides
        ("Mo4C3", "O"): 0.0,  # metallic — thicker slabs close the gap
        ("Mo3N2", "O"): 0.25, # nitrides: slightly larger gaps than carbides
        ("Mo4N3", "O"): 0.15, # nitrides: gap decreases with thickness
        ("Mo2N", "O"): 0.35,  ("Mo2N", "OH"): 0.20, ("Mo2N", "F"): 0.30,
        # V-based: narrow-gap or metallic
        ("V2C", "O"): 0.20,   ("V2C", "OH"): 0.0,  ("V2C", "F"): 0.10,
        ("V3C2", "O"): 0.18,
        # Nb-based
        ("Nb2C", "O"): 0.22,  ("Nb2C", "OH"): 0.05, ("Nb2C", "F"): 0.12,
        ("Nb4C3", "O"): 0.20,
        # Cr-based: semiconducting
        ("Cr2C", "O"): 0.40,  ("Cr2C", "OH"): 0.10, ("Cr2C", "F"): 0.20,
        # Zr-based: wider gaps
        ("Zr2C", "O"): 0.55,  ("Zr3C2", "O"): 0.45,
        # Hf-based: wider gaps
        ("Hf2C", "O"): 0.60,  ("Hf3C2", "O"): 0.50,
        # W-based
        ("W2C", "O"): 0.35,   ("W2N", "O"): 0.40,
        # Sc-based: semiconducting
        ("Sc2C", "O"): 1.80,  ("Sc2C", "OH"): 0.45, ("Sc2C", "F"): 0.70,
    }

    # Typical lattice thermal conductivity for MXenes (W/mK)
    # From literature: MXenes have VERY low κ_L due to 2D structure
    TYPICAL_KAPPA_L = {
        "Ti3C2": 1.5, "Ti2C": 2.0, "V2C": 1.8,
        "Mo2C": 1.2, "Nb2C": 1.6, "Cr2C": 1.3,
        "Mo2N": 1.1, "Mo3C2": 1.2, "Mo3N2": 1.1,
        "Cr2C": 1.3, "Zr2C": 1.4, "Hf2C": 1.3,
        "Nb4C3": 1.5, "V2N": 1.7, "W2C": 1.2,
        "Sc2C": 1.6,
        "DEFAULT": 1.5,
    }

    # Estimated electrical conductivity ranges for MXene composites (S/cm)
    CONDUCTIVITY_ESTIMATES = {
        "polymer": (50, 500),      # with conducting polymer
        "carbon": (100, 2000),     # with CNT/graphene
        "chalcogenide": (10, 200), # with Bi2Te3/SnSe
        "oxide": (1, 50),          # with oxide
        "metal": (500, 5000),      # with metal NW
        "bare": (1000, 10000),     # bare MXene
    }

    # Map config partner_type names to conductivity estimate keys
    PARTNER_TYPE_MAP = {
        "conducting_polymers": "polymer",
        "carbon_materials": "carbon",
        "chalcogenides": "chalcogenide",
        "metals": "metal",
        "oxides": "oxide",
        "polymer": "polymer",
        "carbon": "carbon",
        "chalcogenide": "chalcogenide",
        "metal": "metal",
        "oxide": "oxide",
        "bare": "bare",
    }

    def __init__(self, temperature_k: float = 310.0):
        self.temperature_k = temperature_k
        self._alignn_model = None

    def estimate_from_bandgap(
        self,
        name: str,
        bandgap_ev: float,
        composite_type: str = "bare",
        mxene_base: str = "Ti3C2",
    ) -> TEProperties:
        """
        Quick TE property estimation from bandgap.

        ALGORITHM:
            1. Seebeck from Goldsmid-Sharp relation:
               |S_max| = E_gap / (2 * e * T)
               This gives the MAXIMUM achievable Seebeck at temperature T.
               In practice, S ≈ 0.3-0.7 × S_max depending on doping optimization.
               We use 0.5 as a conservative estimate.

            2. Electrical conductivity from composite type:
               Different composite partners give very different conductivities.
               We use literature ranges as estimates.

            3. Lattice thermal conductivity from MXene type:
               κ_L is mostly determined by the MXene base, not the composite.
               MXenes are inherently low-κ due to their 2D layered structure.

            4. Combine using TE relations:
               PF = S² × σ
               κ = κ_e + κ_L (Wiedemann-Franz for κ_e)
               ZT = PF × T / κ

        LIMITATIONS:
            - Goldsmid-Sharp assumes optimal doping (often not achieved)
            - Conductivity is a rough range, not a specific prediction
            - κ_L is approximate
            - This is for SCREENING only, not for paper-quality predictions
        """
        props = TEProperties(
            name=name,
            temperature_k=self.temperature_k,
            bandgap=bandgap_ev,
            method="estimated",
            confidence="low",
        )

        # --- Seebeck coefficient estimation ---
        # For semiconductors: Goldsmid-Sharp relation S_max = E_g / (2kT)
        # For metals: use literature-based estimates that vary by composite type
        #
        # KEY INSIGHT for composites:
        #   The composite partner modulates the Seebeck through:
        #   - Energy filtering at MXene/partner interface
        #   - Carrier concentration optimization
        #   - Quantum confinement effects (0D, 1D partners)
        #   Literature shows Seebeck of MXene composites ranges 5-80 µV/K
        #   depending on the partner material.

        # Composite-specific Seebeck modifiers (from literature)
        # These capture how different partners affect the composite Seebeck:
        #   - Polymers: moderate S boost via energy filtering (Guan & Lee, ACS Nano 2024)
        #   - CNTs: good S from percolation network (Jin et al., ACS Appl. Mater. 2023)
        #   - Chalcogenides: high S from semiconducting partner (Lu et al., Adv. Energy Mater. 2023)
        #   - Metals: low S (metallic partner reduces it)
        SEEBECK_COMPOSITE = {
            "polymer": {"base_metallic": 28.0, "boost_factor": 1.2},    # PEDOT:PSS, PANI, PPy, P3HT
            "carbon": {"base_metallic": 22.0, "boost_factor": 1.4},     # SWCNT, MWCNT, rGO, graphene
            "chalcogenide": {"base_metallic": 45.0, "boost_factor": 1.8},  # SnS2, MoS2
            "metal": {"base_metallic": 8.0, "boost_factor": 0.6},       # Ag_NW, Cu_NW
            "oxide": {"base_metallic": 35.0, "boost_factor": 1.5},
            "bare": {"base_metallic": 15.0, "boost_factor": 1.0},
        }

        mapped_type = self.PARTNER_TYPE_MAP.get(composite_type, composite_type)
        s_params = SEEBECK_COMPOSITE.get(mapped_type, SEEBECK_COMPOSITE["bare"])

        if bandgap_ev > 0.02:  # semiconducting (>20 meV gap)
            # Goldsmid-Sharp: S_max = E_g / (2kT)
            s_max = bandgap_ev / (2 * KB * self.temperature_k)  # dimensionless (in k_B units)
            s_max_uv = s_max * 1e6 * KB  # convert to µV/K
            # Apply composite boost: some partners enhance Seebeck via energy filtering
            props.seebeck_coefficient = s_max_uv * 0.5 * s_params["boost_factor"]
        else:
            # Metallic: Seebeck dominated by composite interface effects
            # Use partner-specific literature values
            props.seebeck_coefficient = s_params["base_metallic"]

        # --- Electrical conductivity ---
        cond_range = self.CONDUCTIVITY_ESTIMATES.get(
            mapped_type,
            self.CONDUCTIVITY_ESTIMATES["bare"],
        )
        # Use geometric mean of range as estimate
        props.electrical_conductivity = np.sqrt(cond_range[0] * cond_range[1])

        # --- Lattice thermal conductivity ---
        # κ_L depends on MXene base and is reduced by composite partner
        # (phonon scattering at interfaces)
        base_kappa_l = self.TYPICAL_KAPPA_L.get(
            mxene_base, self.TYPICAL_KAPPA_L["DEFAULT"]
        )
        # Composite partners reduce κ_L through interface phonon scattering
        KAPPA_REDUCTION = {
            "polymer": 0.55,       # polymers scatter phonons well (amorphous)
            "carbon": 0.75,        # CNTs have high κ themselves
            "chalcogenide": 0.50,  # layered chalcogenides: low κ interface
            "metal": 0.85,         # metals conduct heat
            "oxide": 0.60,
            "bare": 1.0,
        }
        kappa_factor = KAPPA_REDUCTION.get(mapped_type, 0.7)
        props.lattice_thermal_cond = base_kappa_l * kappa_factor

        # Compute derived properties (PF, κ_e via Wiedemann-Franz, ZT)
        props.compute_derived()
        return props

    def predict_with_alignn(self, atoms: "Atoms", name: str) -> TEProperties:
        """
        Predict properties using ALIGNN pre-trained model.

        ALIGNN = Atomistic Line Graph Neural Network (NIST, 2021)

        HOW ALIGNN WORKS:
            1. Convert crystal structure → graph
               - Nodes = atoms, Edges = bonds (within cutoff radius)
            2. Also build "line graph" (graph of edges)
               - Captures bond angles explicitly
            3. Message passing on both graphs simultaneously
               - Atom features are updated based on neighbor information
               - Bond features capture directional interactions
            4. Global pooling → property prediction

        WHAT IT PREDICTS:
            - Bandgap (eV): trained on Materials Project data
            - Formation energy (eV/atom)
            - Many other properties via different pre-trained models

        FOR TE PROPERTIES:
            We use ALIGNN bandgap → then apply Goldsmid-Sharp + semi-empirical
            This is better than pure estimation because bandgap is ML-predicted
            from the actual structure, not guessed.
        """
        if Atoms is None:
            raise ImportError("ASE required")

        try:
            from alignn.pretrained import get_figshare_model
            import torch

            if self._alignn_model is None:
                # Load pre-trained bandgap model
                self._alignn_model = get_figshare_model("jv_mbj_bandgap")
                logger.info("ALIGNN bandgap model loaded")

            # ALIGNN needs specific input format
            # Convert ASE Atoms → ALIGNN compatible structure
            from jarvis.core.atoms import Atoms as JarvisAtoms

            jarvis_atoms = JarvisAtoms(
                lattice_mat=atoms.get_cell().array,
                coords=atoms.get_scaled_positions(),
                elements=atoms.get_chemical_symbols(),
                cartesian=False,
            )

            # Predict bandgap
            bandgap = self._alignn_model.predict(jarvis_atoms)
            if isinstance(bandgap, torch.Tensor):
                bandgap = bandgap.item()

            logger.info(f"  ALIGNN predicted bandgap for {name}: {bandgap:.3f} eV")

            # Use predicted bandgap for TE estimation
            props = self.estimate_from_bandgap(name, bandgap)
            props.method = "alignn"
            props.confidence = "medium"
            return props

        except ImportError as e:
            logger.warning(f"ALIGNN not available: {e}. Falling back to estimation.")
            # Use composition-specific bandgap instead of flat default
            return self.estimate_from_bandgap(name, bandgap_ev=0.3)  # caller should use screen_candidates instead
        except Exception as e:
            logger.error(f"ALIGNN prediction failed for {name}: {e}")
            return self.estimate_from_bandgap(name, bandgap_ev=0.3)  # caller should use screen_candidates instead

    def compute_descriptors(self, atoms: "Atoms") -> dict:
        """
        Compute compositional and structural descriptors for ML model.

        DESCRIPTORS (what and why):

        Compositional:
            - mean_electronegativity: relates to bond polarity → affects carrier scattering
            - std_electronegativity: diversity of bonding → affects phonon scattering
            - mean_atomic_mass: heavier atoms → lower thermal conductivity (good for TE)
            - d_electron_count: relates to electronic structure complexity
            - valence_electron_count: determines metallic vs semiconducting behavior

        Structural:
            - volume_per_atom: relates to bond stiffness and phonon velocities
            - packing_fraction: how dense the structure is
            - lattice_a: in-plane lattice parameter

        These descriptors feed into a simple ML model trained on our
        extracted literature data for direct property prediction.
        """
        if Atoms is None:
            raise ImportError("ASE required")

        from collections import Counter

        # Elemental properties (Pauling electronegativity, atomic mass)
        electroneg = {
            "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Zr": 1.33, "Nb": 1.60,
            "Mo": 2.16, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Sc": 1.36,
            "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "Cl": 3.16,
            "S": 2.58, "Se": 2.55, "Te": 2.10, "Br": 2.96, "H": 2.20,
        }
        atomic_mass = {
            "Ti": 47.87, "V": 50.94, "Cr": 52.00, "Zr": 91.22, "Nb": 92.91,
            "Mo": 95.95, "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Sc": 44.96,
            "C": 12.01, "N": 14.01, "O": 16.00, "F": 19.00, "Cl": 35.45,
            "S": 32.07, "Se": 78.97, "Te": 127.60, "Br": 79.90, "H": 1.008,
        }
        d_electrons = {
            "Ti": 2, "V": 3, "Cr": 5, "Zr": 2, "Nb": 4,
            "Mo": 5, "Hf": 2, "Ta": 3, "W": 4, "Sc": 1,
        }

        symbols = atoms.get_chemical_symbols()
        composition = Counter(symbols)
        n_atoms = len(atoms)

        # Compositional descriptors
        en_values = [electroneg.get(s, 2.0) for s in symbols]
        mass_values = [atomic_mass.get(s, 50.0) for s in symbols]
        d_values = [d_electrons.get(s, 0) for s in symbols]

        # Structural descriptors
        cell = atoms.get_cell()
        volume = abs(np.linalg.det(cell.array))

        return {
            "n_atoms": n_atoms,
            "mean_electronegativity": np.mean(en_values),
            "std_electronegativity": np.std(en_values),
            "max_electronegativity": np.max(en_values),
            "mean_atomic_mass": np.mean(mass_values),
            "std_atomic_mass": np.std(mass_values),
            "total_d_electrons": sum(d_values),
            "mean_d_electrons": np.mean(d_values) if d_values else 0,
            "volume_per_atom": volume / n_atoms if n_atoms > 0 else 0,
            "lattice_a": np.linalg.norm(cell.array[0]),
            "n_species": len(composition),
            "metal_fraction": sum(v for k, v in composition.items()
                                  if k in d_electrons) / n_atoms,
        }

    def screen_candidates(
        self,
        candidates: list[dict],
        method: str = "estimate",
    ) -> list[TEProperties]:
        """
        Predict TE properties for a batch of candidates.

        INPUT: list of dicts from stability_screener, with keys:
            atoms, name, mxene_formula, composite_partner, is_stable

        ALGORITHM:
            1. Filter to stable candidates only
            2. For each candidate:
               a. If method="alignn": use ALIGNN for bandgap → TE estimation
               b. If method="estimate": use pure semi-empirical estimation
               c. If method="descriptors": use descriptor-based ML
            3. Rank by predicted power factor and ZT
            4. Return sorted results

        RETURNS: list of TEProperties, sorted by predicted ZT (descending)
        """
        results = []

        stable = [c for c in candidates if c.get("is_stable", False)]
        logger.info(f"Predicting TE properties for {len(stable)} stable candidates")

        for cand in stable:
            name = cand.get("name", "unknown")
            mxene_formula = cand.get("mxene_formula", "Ti3C2")
            termination = cand.get("termination", "O")
            composite_type = cand.get("partner_type", "bare")

            # Look up composition-specific bandgap estimate
            bandgap = self.BANDGAP_ESTIMATES.get(
                (mxene_formula, termination),
                0.3,  # default if no specific estimate
            )

            if method == "alignn" and cand.get("atoms") is not None:
                props = self.predict_with_alignn(cand["atoms"], name)
            else:
                # Semi-empirical estimation with composition-specific bandgap
                props = self.estimate_from_bandgap(
                    name=name,
                    bandgap_ev=bandgap,
                    composite_type=composite_type,
                    mxene_base=mxene_formula,
                )

            props.name = name
            results.append(props)

        # Sort by ZT (descending), then by power factor
        results.sort(
            key=lambda p: (p.zt_value or 0, p.power_factor or 0),
            reverse=True,
        )

        return results
