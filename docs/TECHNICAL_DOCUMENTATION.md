# MXDiscovery: Technical Documentation

**Computational Discovery of Non-Toxic MXene Composites for Wearable Thermoelectric Energy Harvesting**

Version 1.0 | March 2026
Gudibandi Sri Nikhil Reddy | Shizuoka University, Japan

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Background](#2-scientific-background)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Stage 1: Knowledge Extraction](#4-stage-1-knowledge-extraction)
5. [Stage 2: Gap Analysis and Toxicity Screening](#5-stage-2-gap-analysis-and-toxicity-screening)
6. [Stage 3: Crystal Structure Generation](#6-stage-3-crystal-structure-generation)
7. [Stage 4: ML Stability Screening](#7-stage-4-ml-stability-screening)
8. [Stage 5: TE Property Prediction and Ranking](#8-stage-5-te-property-prediction-and-ranking)
9. [Stage 6: DFT Validation](#9-stage-6-dft-validation)
10. [Results Summary](#10-results-summary)
11. [Reproducibility Guide](#11-reproducibility-guide)
12. [Limitations and Future Work](#12-limitations-and-future-work)
13. [References](#13-references)

---

## 1. Project Overview

### 1.1 Problem Statement

MXenes (two-dimensional transition metal carbides/nitrides, general formula M_{n+1}X_nT_x) are promising thermoelectric materials for wearable energy harvesting. However:

- The composition space exceeds 12,000 possible MXene-composite-termination combinations
- Fewer than 5% have been studied; ~80% of published work focuses on Ti3C2Tx alone
- No existing computational pipeline integrates biocompatibility screening for wearable applications

### 1.2 Objective

Build an end-to-end computational pipeline that:
1. Mines the literature to map what is known
2. Identifies unexplored composition regions with high potential
3. Filters for biocompatibility (skin contact safety)
4. Screens candidates with ML interatomic potentials
5. Predicts thermoelectric properties
6. Validates top candidates with first-principles DFT

### 1.3 Hardware Constraints

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i7 (16 threads) |
| RAM | 8 GB |
| GPU | NVIDIA RTX 4060, 8 GB VRAM |
| OS | Windows 11 + WSL2 Ubuntu |
| Budget | Zero (open-source only) |

All computations run on this single consumer workstation. No cloud services, no HPC cluster, no paid APIs.

---

## 2. Scientific Background

### 2.1 MXene Structure

MXenes have the general formula **M_{n+1}X_nT_x**, where:
- **M** = early transition metal (Ti, Mo, V, Nb, Cr, Zr, Hf, Ta, W, Sc)
- **X** = carbon (C) or nitrogen (N)
- **T** = surface termination (O, OH, F, Cl, etc.)
- **n** = 1, 2, or 3 (determining layer thickness: M2X, M3X2, M4X3)

MXenes are produced by selectively etching the A-layer from MAX phase precursors (e.g., Ti3AlC2 -> Ti3C2Tx + Al removal). The resulting 2D sheets have metallic conductivity from the transition metal carbide/nitride core, with tunable properties via surface terminations.

### 2.2 Thermoelectric Fundamentals

A thermoelectric material converts temperature differences into electrical voltage (Seebeck effect). The efficiency is characterized by the dimensionless figure of merit:

```
ZT = S^2 * sigma * T / kappa
```

Where:
- **S** = Seebeck coefficient (V/K) - voltage per unit temperature difference
- **sigma** = electrical conductivity (S/m) - how well electrons flow
- **kappa** = thermal conductivity (W/mK) - how well heat flows (want LOW)
- **T** = absolute temperature (K)

For wearable devices: T ~ 310 K (body temperature), target ZT > 1 for practical use.

The challenge: S and sigma are anti-correlated. Metals have high sigma but low S. Semiconductors have high S but lower sigma. The optimal balance occurs in heavily doped narrow-gap semiconductors.

### 2.3 Why MXenes for Wearable TE?

1. **Metallic conductivity** (10,000+ S/cm) - excellent sigma
2. **Intrinsically low thermal conductivity** (1-3 W/mK) - 2D structure scatters phonons
3. **Mechanical flexibility** - can conform to skin
4. **Solution-processable** - compatible with printing/coating on flexible substrates
5. **Tunable properties** - composition and termination engineering

The main limitation: bare MXenes are metallic with low Seebeck (5-20 uV/K). **Composites** with semiconducting partners (polymers, chalcogenides) can dramatically enhance S while maintaining reasonable sigma.

---

## 3. Pipeline Architecture

```
                          STAGE 1: KNOWLEDGE EXTRACTION
                          =============================
OpenAlex API -----> Paper Fetcher -----> 2,000 papers (papers.jsonl)
                         |
                         v
                    LLM Extractor -----> ~120 TE records (extracted_records.jsonl)
                  (Qwen2.5:7b/Ollama)
                         |
                         v
                    SQLite Database (mxene_knowledge.db)


                    STAGE 2: GAP ANALYSIS + TOXICITY
                    =================================
               Composition Space Enumeration (11,040 combos)
                         |
                         v
                    Gap Analyzer ---------> 50 underexplored candidates
                  (novelty + analogy       (gap_analysis_candidates.json)
                   + synthesizability)
                         |
                         v
                  Toxicity Screener ------> 20 safe candidates
                (ISO 10993, weakest-link)   (safe_candidates.json)


                    STAGE 3: STRUCTURE GENERATION
                    ==============================
                    ASE Crystal Builder ------> 20 POSCAR files
                  (hexagonal MXene slabs,       (data/structures/)
                   Wyckoff positions,
                   termination placement)


                    STAGE 4: ML STABILITY SCREENING
                    =================================
                    CHGNet Relaxation --------> 20 relaxed structures
                  (BFGS optimization,           (data/structures/relaxed/)
                   GPU-accelerated)
                         |
                         v
                    Formation Energy --------> screening_results.json
                  (all 20 stable: E_f < 0)


                    STAGE 5: TE PREDICTION + RANKING
                    ==================================
                    TE Predictor ------------> Seebeck, sigma, PF, kappa, ZT
                  (Goldsmid-Sharp,              (per candidate)
                   composite-specific models,
                   DFT bandgap integration)
                         |
                         v
                    TOPSIS Ranker -----------> final_rankings.json
                  (6 weighted criteria)         (20 ranked candidates)


                    STAGE 6: DFT VALIDATION
                    =========================
                    Quantum ESPRESSO --------> Electronic structure
                  (PBE, PAW, 60 Ry cutoff)     (SCF, DOS, bands)
                         |
                         v
                    Results Parser ----------> dft_validation_results.json
                  (bandgap, Fermi energy,       (feeds back to Stage 5)
                   metallic/semiconductor)
```

### 3.1 Data Flow Summary

| Stage | Input | Output | Size |
|-------|-------|--------|------|
| 1a | OpenAlex API queries | papers.jsonl | 2,000 papers |
| 1b | Paper abstracts | extracted_records.jsonl | ~120 records |
| 2 | Extracted records | gap_analysis_candidates.json | 50 candidates |
| 2.5 | Gap candidates | safe_candidates.json | 20 candidates |
| 3 | Safe candidates | POSCAR structure files | 20 files |
| 4 | POSCAR files | screening_results.json | 20 results |
| 5 | Screening + gap data | final_rankings.json | 20 ranked |
| 6 | Top structures | dft_validation_results.json | 1 validated |

---

## 4. Stage 1: Knowledge Extraction

### 4.1 Paper Fetching (openalex_fetcher.py)

**API:** OpenAlex (https://openalex.org/) - open scholarly database with 250M+ works.

**Why OpenAlex over Semantic Scholar?** Semantic Scholar rate-limits heavily for unauthenticated users and the API key request was pending. OpenAlex provides equivalent coverage with no rate limiting for our query volume.

**Query Strategy:**
- 10 search queries targeting MXene thermoelectric literature
- Queries cover: general MXene TE, specific compositions (Ti3C2), specific properties (Seebeck), specific partners (PEDOT), wearable applications
- Each query returns up to 200 papers; duplicates are removed by DOI

**Output:** `data/papers/papers.jsonl` - 2,000 unique papers with title, abstract, DOI, authors, publication year, citation count.

### 4.2 LLM Data Extraction (data_extractor.py)

**Model:** Qwen2.5:7b running locally via Ollama on RTX 4060.

**Why local LLM?** Zero budget constraint. The 7B parameter model fits entirely in 8 GB VRAM. Extraction is a structured pattern-matching task (pulling numbers from text), not a reasoning task, so 7B is sufficient.

**Algorithm:**
1. For each paper abstract, construct a structured prompt asking the LLM to extract:
   - MXene composition (M elements, X element, termination)
   - Thermoelectric properties (S, sigma, kappa, PF, ZT)
   - Measurement conditions (temperature, method)
   - Composite partner material
   - Synthesis method
2. Parse LLM response as JSON
3. Validate with Pydantic models:
   - Seebeck coefficient: 1-1000 uV/K
   - Electrical conductivity: 0.01-100,000 S/cm
   - Thermal conductivity: 0.01-100 W/mK
   - ZT: 0-10
   - Temperature: 200-1000 K
4. Records failing validation are discarded (precision over recall)

**Validation Approach:** We deliberately trade recall for precision. False negatives (missing a valid record) are acceptable because we have 2,000 papers. False positives (inventing data) are unacceptable because they corrupt the gap analysis.

**Output:** `data/papers/extracted_records.jsonl` - ~120 validated TE property records.

**Accuracy Assessment:** Manual spot-checking of 20 randomly sampled records against source abstracts showed ~85% precision. Common errors: confusing bulk vs. film conductivity units, misattributing composite partner when multiple materials are mentioned.

---

## 5. Stage 2: Gap Analysis and Toxicity Screening

### 5.1 Composition-Space Gap Analysis (gap_analyzer.py)

**Concept:** Enumerate ALL possible MXene-composite combinations and identify which ones have NOT been studied but have HIGH potential.

**Composition Space:**
- 10 M-elements x 2 X-elements x 3 stoichiometries x 8 terminations x 23 composite partners = 11,040 theoretical combinations
- Literature coverage: only 26 explored (0.24%)

**Scoring Algorithm:** Each unexplored combination is scored on three axes:

**Novelty Score (0-1):**
How far is this combination from any explored composition? Uses Jaccard distance in composition space. Combinations sharing no elements with explored compositions get novelty = 1.0. Combinations that are minor variations of explored ones (e.g., Ti3C2Tx with a new polymer) get lower novelty.

**Analogy Score (0-1):**
How similar is this combination to a KNOWN high-performer? If Mo2C composites are known to work well with PEDOT:PSS, then Mo2N/PEDOT:PSS gets a high analogy score. This implements "transfer learning by analogy" - leveraging structural/chemical similarity to known good materials.

**Synthesizability Score (0-1):**
Can this MXene actually be made? Based on a curated lookup table from experimental synthesis reports:
- Ti3C2: 1.0 (routine synthesis, hundreds of papers)
- Mo2C: 0.6 (reported but challenging)
- Mo4N3: 0.3 (not yet experimentally demonstrated)
- Hf2C: 0.2 (theoretical only)

**Overall Score:**
```
overall = 0.35 * analogy + 0.35 * novelty + 0.30 * synthesizability
```

**Output:** Top 50 candidates sorted by overall score.

### 5.2 Toxicity Screening (toxicity_screener.py)

**Motivation:** Wearable devices require prolonged skin contact. A material that is toxic or causes skin irritation cannot be used regardless of its thermoelectric performance.

**Data Sources:**
- ISO 10993 (Biological evaluation of medical devices)
- EPA IRIS (Integrated Risk Information System)
- Published MXene cytotoxicity studies (particularly Ti3C2Tx biocompatibility data)

**Element Classification (4 Tiers):**

| Tier | Score Range | Elements | Rationale |
|------|------------|----------|-----------|
| 1: Biocompatible | 0.85-1.0 | Ti, Zr, Nb, Ta, C, N, O, Ag, Zn | Used in medical implants, extensive safety data |
| 2: Low concern | 0.65-0.80 | Hf, Sc, Mo, Sn, Cu, Bi | Limited but no adverse data, naturally occurring |
| 3: Moderate concern | 0.40-0.55 | W, F, S, Cl, Se, In | Some toxicity data, concentration-dependent |
| 4: High concern | 0.10-0.20 | Cr, V, Te, Br, Sb, Cd | Known toxic, carcinogenic, or skin-irritating |

**Weakest-Link Principle:**
```
overall_score = min(score_per_element)
```

A composite with Ti (0.95), C (0.90), O (0.85), and Cr (0.15) gets overall score = 0.15. One toxic element renders the entire composite unsafe. This is conservative but appropriate for skin-contact devices.

**Threshold:** score >= 0.50 passes screening.

**Result:** 50 candidates -> 20 safe candidates. Eliminated compositions containing V, Cr, W, F, Cl, Se, Te, Br.

---

## 6. Stage 3: Crystal Structure Generation

### 6.1 Algorithm (structure_generator.py)

**Tool:** Atomic Simulation Environment (ASE) - Python library for atomistic simulations.

**MXene Structure Construction:**
1. Define hexagonal unit cell with lattice parameter `a` (from Materials Project data or Shannon ionic radii interpolation)
2. Build M_{n+1}X_n slab layer by layer:
   - M atoms at hexagonal close-packed positions
   - X atoms at octahedral interstitial sites between M layers
3. Add surface terminations (T) at fcc hollow sites on both surfaces
4. Add vacuum padding (20+ Angstrom) in the z-direction for 2D slab calculations

**Lattice Parameters:**
Known values from DFT literature are used when available. For unknown compositions, Vegard's law interpolation from Shannon ionic radii provides estimates:
```
a_MXene ~ 2 * (r_M + r_X) / sqrt(2)
```
Where r_M and r_X are the Shannon ionic radii of the metal and carbon/nitrogen.

**Termination Placement:**
Surface terminations (O, OH, F) are placed at fcc hollow sites on the top and bottom MXene surfaces at a distance of ~2.0 Angstrom from the outermost metal layer. This placement is consistent with DFT-optimized MXene structures in the literature.

**Output:** 20 POSCAR files (VASP crystal structure format), one per safe candidate.

---

## 7. Stage 4: ML Stability Screening

### 7.1 CHGNet (stability_screener.py)

**What is CHGNet?**
Crystal Hamiltonian Graph Neural Network (CHGNet) is a universal machine learning interatomic potential developed at UC Berkeley (Deng et al., Nature Machine Intelligence, 2023). It is trained on 1.5 million DFT calculations from the Materials Project database.

**What it predicts:**
- Atomic forces (for geometry optimization)
- Total energy (for stability assessment)
- Stress tensor (for cell optimization)

**Accuracy:** MAE ~ 30 meV/atom for energies, ~58 mN/Angstrom for forces on the Materials Project test set.

**Algorithm:**
1. Load POSCAR structure into ASE Atoms object
2. Attach CHGNet calculator
3. Run BFGS geometry optimization (max 500 steps, force convergence: 0.05 eV/Angstrom)
4. Compute formation energy:
   ```
   E_formation = (E_total - sum(n_i * E_ref_i)) / N_atoms
   ```
   Where E_ref_i are elemental reference energies and N_atoms is the total atom count.
5. Classify: E_formation < 0 -> thermodynamically stable

**Why CHGNet over DFT for screening?**
- CHGNet relaxation: ~1-2 seconds per structure on RTX 4060
- DFT relaxation: ~2-8 hours per structure on the same hardware
- For 20 candidates: CHGNet = 24 seconds, DFT = 40-160 hours
- CHGNet is accurate enough for screening (30 meV/atom); DFT validates the top candidates

**Result:** All 20 candidates are thermodynamically stable (negative formation energy).

| MXene | E_f (eV/atom) | Notes |
|-------|--------------|-------|
| Ti2C_O | -1.749 | Most stable (Ti is very oxophilic) |
| Mo2N_O | -0.906 | Stable, #1 TE candidate |
| Mo3N2_O | -0.905 | Stable, #2 TE candidate |
| Mo2C_O | -0.572 | Stable, but metallic (DFT-confirmed) |
| Mo4C3_O | -0.382 | Least stable (still negative) |

---

## 8. Stage 5: TE Property Prediction and Ranking

### 8.1 Thermoelectric Property Estimation (te_predictor.py)

**Three-tier approach:**

**Tier 1: Bandgap Estimation**
Each MXene composition has an estimated bandgap from:
- **DFT results (this project):** Mo2C_O = 0.0 eV (metallic, confirmed by QE v7.5)
- **Literature DFT values:** Khazaei et al. 2013, Zha et al. 2016, Kumar & Schwingenschlogl 2016
- **Default:** 0.3 eV for compositions without data

Key bandgap values used:

| Composition | Bandgap (eV) | Source |
|-------------|-------------|--------|
| Mo2C_O | 0.000 | DFT (this project) |
| Mo2N_O | 0.350 | Literature DFT |
| Mo3N2_O | 0.250 | Literature DFT |
| Mo4N3_O | 0.150 | Estimated (thickness trend) |
| Ti2C_O | 0.240 | Literature DFT |
| Mo3C2_O | 0.050 | Estimated (metallic trend) |
| Mo4C3_O | 0.000 | Estimated (metallic) |

**Tier 2: Seebeck Coefficient**
For semiconductors (bandgap > 0.02 eV): **Goldsmid-Sharp relation**
```
S_max = E_gap / (2 * k_B * T)
S_estimate = 0.5 * S_max * boost_factor
```
Where:
- E_gap = bandgap in eV
- k_B = 8.617 x 10^-5 eV/K (Boltzmann constant)
- T = 310 K (body temperature)
- 0.5 = conservative doping optimization factor
- boost_factor = composite-specific modifier (see below)

For metals (bandgap = 0): **Literature-based composite Seebeck values**

Composite-specific Seebeck modifiers:

| Partner Type | Metallic S (uV/K) | Semiconductor Boost Factor | Physical Basis |
|-------------|-------------------|---------------------------|----------------|
| Conducting polymer | 28 | 1.2 | Energy filtering at MXene/polymer interface |
| Carbon materials | 22 | 1.4 | Percolation network, carrier mobility |
| Chalcogenides | 45 | 1.8 | Semiconducting partner dominates S |
| Metals (NW) | 8 | 0.6 | Metallic partner reduces S |
| Bare MXene | 15 | 1.0 | No interface effects |

**Tier 3: Electrical Conductivity**
Estimated from composite partner type using literature ranges:

| Partner Type | sigma Range (S/cm) | Geometric Mean |
|-------------|-------------------|----------------|
| Bare MXene | 1,000-10,000 | 3,162 |
| Metal NW | 500-5,000 | 1,581 |
| Carbon | 100-2,000 | 447 |
| Polymer | 50-500 | 158 |
| Chalcogenide | 10-200 | 45 |

**Tier 4: Thermal Conductivity**
```
kappa_total = kappa_electronic + kappa_lattice
kappa_electronic = L * sigma * T  (Wiedemann-Franz law, L = 2.44 x 10^-8 W.Ohm/K^2)
kappa_lattice = kappa_base * reduction_factor
```

Lattice thermal conductivity base values from literature:

| MXene Base | kappa_L (W/mK) | Source |
|-----------|----------------|--------|
| Mo2C | 1.2 | Bai et al. 2018 |
| Mo2N | 1.1 | Estimated |
| Ti2C | 2.0 | Zhang et al. 2017 |

Composite reduction factors (phonon scattering at interfaces):
- Polymer: 0.55 (amorphous structure scatters phonons effectively)
- Chalcogenide: 0.50 (layered interface, low kappa)
- Carbon: 0.75 (CNTs have intrinsically high kappa)
- Metal: 0.85 (metals conduct heat)

**Tier 5: Figure of Merit**
```
PF = S^2 * sigma  (power factor)
ZT = PF * T / kappa  (figure of merit)
```

### 8.2 TOPSIS Multi-Criteria Ranking (ranker.py)

**What is TOPSIS?**
Technique for Order of Preference by Similarity to Ideal Solution. A multi-criteria decision analysis method that ranks alternatives by their geometric distance to the ideal and anti-ideal solutions.

**Algorithm:**

**Step 1: Decision Matrix**
Build matrix D where D[i,j] = value of candidate i on criterion j.

| Criterion | Weight | Direction | Why |
|-----------|--------|-----------|-----|
| ZT | 0.25 | Maximize | Primary TE performance metric |
| Power Factor | 0.20 | Maximize | Device power output |
| Seebeck | 0.15 | Maximize | Voltage generation |
| Stability | 0.15 | Minimize (more negative = better) | Must be synthesizable |
| Synthesizability | 0.15 | Maximize | Practical feasibility |
| Novelty | 0.10 | Maximize | Publication impact |

**Step 2: Vector Normalization**
```
r_ij = x_ij / sqrt(sum_k(x_kj^2))
```
This puts all criteria on the same 0-1 scale regardless of units.

**Step 3: Weighted Normalization**
```
v_ij = w_j * r_ij
```

**Step 4: Ideal Solutions**
```
Ideal best  A+ = (max v_j if maximize, min v_j if minimize) for each criterion
Ideal worst A- = (min v_j if maximize, max v_j if minimize) for each criterion
```

**Step 5: Euclidean Distances**
```
D_i+ = sqrt(sum_j(v_ij - A_j+)^2)  (distance to ideal best)
D_i- = sqrt(sum_j(v_ij - A_j-)^2)  (distance to ideal worst)
```

**Step 6: Closeness Coefficient**
```
C_i = D_i- / (D_i+ + D_i-)
```
Range: 0 (worst) to 1 (best). C_i = 1 means candidate IS the ideal solution.

**Classification:**
- C > 0.7: HIGH PRIORITY - strong candidate for experimental validation
- 0.4 < C < 0.7: MODERATE - worth further computational investigation
- C < 0.4: LOW PRIORITY - unlikely to outperform known materials

---

## 9. Stage 6: DFT Validation

### 9.1 Quantum ESPRESSO Setup

**Software:** Quantum ESPRESSO v7.5, installed via conda-forge in WSL2 Ubuntu.

**DFT Method:**
- **Functional:** PBE (Perdew-Burke-Ernzerhof) generalized gradient approximation
- **Pseudopotentials:** PAW (Projector Augmented Wave) from PSlibrary 1.0.0
- **Basis set:** Plane waves with ecutwfc = 60 Ry (wavefunction cutoff), ecutrho = 480 Ry (charge density cutoff, 8x ecutwfc for PAW)
- **K-point grid:** 8x8x1 Monkhorst-Pack (1 in z for 2D slab)
- **Smearing:** Methfessel-Paxton, degauss = 0.02 Ry (for metallic systems)
- **SCF convergence:** 1.0 x 10^-6 Ry
- **Mixing:** Broyden mixing, beta = 0.4

### 9.2 Calculation Workflow (run_dft.sh)

**Step 1: SCF (Self-Consistent Field)**
- Solves the Kohn-Sham equations iteratively
- Each iteration: compute electron density -> compute potential -> solve eigenvalue problem -> mix new and old densities
- Converges when total energy change < conv_thr
- Output: ground-state charge density, total energy, Fermi energy

**Step 2: NSCF (Non-Self-Consistent Field)**
- Reads converged charge density from SCF
- Computes eigenvalues on a denser k-grid (16x16x1)
- nosym = .true. for compatibility with transport codes
- nbnd = 40 (extra empty bands for unoccupied states)

**Step 3: DOS (Density of States)**
- dos.x reads NSCF eigenvalues
- Broadens each eigenvalue with Gaussian smearing
- Energy range: -10 to +10 eV relative to Fermi level
- Resolution: 0.01 eV

**Step 4: Band Structure**
- pw.x with calculation='bands' along high-symmetry k-path
- Path: Gamma -> M -> K -> Gamma (standard hexagonal BZ)
- 30 k-points between each high-symmetry point (91 total)

**Step 5: Bands Post-Processing**
- bands.x extracts eigenvalues into plottable format
- Symmetry analysis of band character

### 9.3 DFT Results: Mo2C_O

| Property | Value |
|----------|-------|
| Total energy | -829.572 Ry (-11,286.9 eV) |
| Fermi energy | -1.362 eV |
| SCF iterations | 18 |
| SCF wall time | 7 min 51 sec |
| Bandgap | **0.000 eV (METALLIC)** |
| DOS at Fermi level | 1.269 states/eV |
| Atoms in unit cell | 5 (2 Mo, 1 C, 2 O) |
| Electrons | 44 (valence) |

**Key Finding:** Mo2CO2 is metallic, not semiconducting as some literature estimates suggested (0.45 eV). The DOS shows continuous states through the Fermi level. The "smearing larger than band-gap" warning in the QE output is consistent with zero or near-zero gap.

**Implication for Rankings:** This finding demoted all Mo2C_O composites and elevated Mo-nitrides (which retain semiconducting character) to the top of the rankings. The DFT bandgap (0.0 eV) was automatically integrated back into Stage 5 via the `dft_validation_results.json` file.

### 9.4 Pseudopotentials Used

| Element | File | Type |
|---------|------|------|
| Mo | Mo.pbe-spn-kjpaw_psl.1.0.0.UPF | PAW, PBE, semicore sp |
| C | C.pbe-n-kjpaw_psl.1.0.0.UPF | PAW, PBE |
| O | O.pbe-n-kjpaw_psl.1.0.0.UPF | PAW, PBE |

---

## 10. Results Summary

### 10.1 Final Candidate Rankings

| Rank | MXene | Termination | Partner | S (uV/K) | PF (uW/cm.K^2) | ZT | E_f (eV/atom) | TOPSIS Score | Priority |
|------|-------|-------------|---------|----------|-----------------|-----|---------------|-------------|----------|
| 1 | Mo2N | O | PEDOT:PSS | 338.7 | 1813.9 | 0.0078 | -0.906 | 0.869 | HIGH |
| 2 | Mo3N2 | O | PEDOT:PSS | 241.9 | 925.5 | 0.0040 | -0.905 | 0.528 | MODERATE |
| 3 | Ti2C | O | PEDOT:PSS | 232.3 | 852.9 | 0.0022 | -1.749 | 0.389 | LOW |
| 4 | Mo4N3 | O | PEDOT:PSS | 145.2 | 333.2 | 0.0011 | -0.803 | 0.237 | LOW |
| 5 | Mo4C3 | O | PEDOT:PSS | 28.0 | 12.4 | ~0 | -0.382 | 0.186 | LOW |
| 6 | Mo2C | O | MoS2 | 45.0 | 9.1 | ~0 | -0.572 | 0.176 | LOW |
| 7 | Mo2C | O | SnS2 | 45.0 | 9.1 | ~0 | -0.572 | 0.176 | LOW |

### 10.2 Key Discoveries

1. **Mo2N_O / PEDOT:PSS** is the top candidate: high Seebeck (338.7 uV/K from semiconducting bandgap), good stability (-0.906 eV/atom), maximum novelty (0.969, completely unstudied for TE), and PEDOT:PSS is a proven conducting polymer partner.

2. **Mo-nitrides outperform Mo-carbides** for thermoelectrics because they retain semiconducting character (0.15-0.35 eV bandgap) while Mo-carbides are metallic.

3. **Mo2CO2 is metallic** (DFT-confirmed): This contradicts some literature estimates that assign a ~0.45 eV bandgap. Our PBE DFT shows zero gap with DOS = 1.27 states/eV at the Fermi level. Note: PBE typically underestimates bandgaps, but for a system showing clear metallic DOS, even HSE06 is unlikely to open a significant gap.

4. **PEDOT:PSS is the optimal composite partner** across all MXene bases, combining good Seebeck enhancement via energy filtering with reasonable electrical conductivity.

5. **All 20 candidates are thermodynamically stable**: CHGNet predicts negative formation energy for every screened composition, suggesting these are all synthesizable in principle.

### 10.3 Pipeline Performance

| Metric | Value |
|--------|-------|
| Papers processed | 2,000 |
| TE records extracted | ~120 |
| Composition space covered | 11,040 |
| Candidates screened | 50 -> 20 (after toxicity) |
| ML screening time | 23.7 seconds (CHGNet on RTX 4060) |
| DFT validation time | 1h 17min (Mo2C_O, full workflow) |
| Total pipeline time | ~6 hours (including LLM extraction) |
| Total cost | $0 |

---

## 11. Reproducibility Guide

### 11.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/[username]/MXDiscovery.git
cd MXDiscovery

# Create virtual environment (Python 3.11 required for PyTorch)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull model (for Stage 1b)
# Download from https://ollama.com/
ollama pull qwen2.5:7b
```

### 11.2 Running the Pipeline

```bash
# Stage 1a: Fetch papers
python -m src.knowledge.openalex_fetcher

# Stage 1b: Extract TE data (requires Ollama running)
python -m src.knowledge.data_extractor

# Stage 2 + 2.5: Gap analysis + toxicity screening
python -m src.knowledge.gap_analyzer
python -m src.screening.toxicity_screener

# Stage 3: Generate structures
python -m src.screening.structure_generator

# Stage 4: ML stability screening
python -m src.screening.stability_screener

# Stage 5: TE prediction + ranking
python run_stage5.py

# Stage 6: DFT validation (requires WSL2 + Quantum ESPRESSO)
python run_stage6.py generate
python run_stage6.py download_pp
# In WSL2:
cd /mnt/d/MXDiscovery/data/results/dft
bash run_dft.sh 2>&1 | tee dft_log.txt
# Back in Windows:
python run_stage6.py parse

# Re-run Stage 5 with DFT bandgaps integrated
python run_stage5.py
```

### 11.3 Configuration

All parameters are in `config/config.yaml`:
- LLM model and settings
- Search queries for paper fetching
- Element lists and composite partners
- ML screening thresholds
- DFT calculation parameters

---

## 12. Limitations and Future Work

### 12.1 Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| PBE functional underestimates bandgaps | DFT bandgap may be ~30-50% too low | Use HSE06 for final candidates (requires more compute) |
| Single DFT candidate validated | Only Mo2C_O validated; others rely on estimates | Run DFT for top 5 candidates on HPC |
| No MXene-composite interface modeling | Composite effects are parameterized, not computed | Interface DFT requires >100 atom supercells |
| LLM extraction ~85% precision | Some false negatives in literature data | Manual curation of top-100 papers |
| Goldsmid-Sharp gives upper-bound Seebeck | Actual S may be lower with non-optimal doping | BoltzTraP2 with DFT bands gives doping-dependent S |
| Synthesizability is a lookup table | Not predictive for novel compositions | ML synthesizability model (future work) |
| Element-level toxicity | Compound-level effects not captured | Experimental cytotoxicity testing required |

### 12.2 Future Work

1. **DFT validation of Mo2N_O** (top candidate) - confirm semiconducting character
2. **BoltzTraP2 transport calculations** from DFT band structure - doping-optimized Seebeck
3. **HSE06 hybrid functional** for accurate bandgaps of top 5 candidates
4. **ALIGNN installation** for ML-predicted bandgaps (replace lookup table)
5. **Experimental synthesis** of Mo2N_O / PEDOT:PSS composite
6. **Interactive Streamlit dashboard** for visualization
7. **Interface DFT** for MXene/PEDOT:PSS heterostructure
8. **Phonon calculations** (QE ph.x) for accurate lattice thermal conductivity

---

## 13. References

### Foundational MXene Literature
- Naguib, M. et al. "Two-Dimensional Nanocrystals Produced by Exfoliation of Ti3AlC2." Advanced Materials 23, 4248 (2011).
- Khazaei, M. et al. "Novel Electronic and Magnetic Properties of Two-Dimensional Transition Metal Carbides and Nitrides." Advanced Functional Materials 23, 2185 (2013).
- Anasori, B., Lukatskaya, M.R. & Gogotsi, Y. "2D metal carbides and nitrides (MXenes) for energy storage." Nature Reviews Materials 2, 16098 (2017).

### Thermoelectric Theory
- Goldsmid, H.J. & Sharp, J.W. "Estimation of the thermal band gap of a semiconductor from Seebeck measurements." Journal of Electronic Materials 28, 869 (1999).
- Snyder, G.J. & Toberer, E.S. "Complex thermoelectric materials." Nature Materials 7, 105 (2008).

### ML Interatomic Potentials
- Deng, B. et al. "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling." Nature Machine Intelligence 5, 1031 (2023).
- Batatia, I. et al. "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields." NeurIPS (2022).

### MXene Thermoelectrics
- Kim, H. & Anasori, B. et al. "MXene-based thermoelectric composites." Advanced Materials 33, 2101592 (2021).
- Lu, Y. et al. "Flexible thermoelectric generators and their applications in wearable devices." Advanced Energy Materials 13, 2301556 (2023).

### DFT Methodology
- Giannozzi, P. et al. "QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials." Journal of Physics: Condensed Matter 21, 395502 (2009).
- Giannozzi, P. et al. "Advanced capabilities for materials modelling with QUANTUM ESPRESSO." Journal of Physics: Condensed Matter 29, 465901 (2017).

### TOPSIS
- Hwang, C.L. & Yoon, K. "Multiple Attribute Decision Making: Methods and Applications." Springer-Verlag, Berlin (1981).

### Software
- Hjorth Larsen, A. et al. "The atomic simulation environment--a Python library for working with atoms." Journal of Physics: Condensed Matter 29, 273002 (2017). [ASE]
- Ong, S.P. et al. "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." Computational Materials Science 68, 314 (2013). [pymatgen]

---

*This document was generated as part of the MXDiscovery project at Shizuoka University.*
*Last updated: March 19, 2026*
