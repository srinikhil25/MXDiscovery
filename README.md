# MXDiscovery

**Computational Pipeline for Discovering Non-Toxic MXene Composites for Wearable Thermoelectric Energy Harvesting**

---

## The Problem

MXenes — two-dimensional transition metal carbides and nitrides — have emerged as promising thermoelectric materials for wearable energy harvesting from body heat. They combine metallic electrical conductivity, intrinsically low thermal conductivity, and mechanical flexibility, all of which are desirable for devices worn on the skin.

However, MXene thermoelectric research faces three critical bottlenecks:

**1. The composition space is vastly underexplored.**
With over 30 experimentally synthesized MXene compositions, 8+ surface termination types, and dozens of viable composite partners (conducting polymers, carbon nanomaterials, chalcogenides, oxides), the total number of possible MXene-composite-termination combinations exceeds 12,000. Fewer than 5% of these have been studied for thermoelectric properties. The field is overwhelmingly concentrated on Ti₃C₂Tₓ (~80% of published work), leaving the majority of the composition space completely blind.

**2. No systematic method exists to identify what hasn't been tried.**
Researchers choose compositions based on intuition, material availability, or incremental variation from prior work. There is no quantitative framework for mapping the explored vs. unexplored regions of the MXene thermoelectric composition space, and no principled way to prioritize which unexplored combinations are most likely to yield high-performing materials.

**3. Biocompatibility is ignored in computational screening.**
For wearable applications requiring prolonged skin contact, the toxicity of constituent elements is a hard constraint, not an afterthought. Several MXene compositions that perform well thermoelectrically (V₂CTₓ, Cr₂TiC₂Tₓ) contain elements with significant toxicity concerns. No existing computational screening pipeline integrates biocompatibility assessment.

## What MXDiscovery Does

MXDiscovery is a six-stage computational pipeline that addresses all three bottlenecks:

### Stage 1 — Knowledge Extraction
Automatically fetches MXene thermoelectric papers from the Semantic Scholar API and extracts structured property data (Seebeck coefficient, electrical conductivity, thermal conductivity, power factor, ZT, synthesis method, composite partner) from abstracts using a local large language model. The extracted data populates a relational database that serves as the foundation for all downstream analysis.

### Stage 2 — Composition-Space Gap Analysis
Enumerates the full theoretical MXene-composite-termination space and cross-references it against the literature database to identify unexplored regions. Each unexplored combination is scored on three axes:

- **Analogy** — How similar is this combination to known high-performing thermoelectrics? Combinations that share a metal element or composite partner with a proven performer score higher.
- **Novelty** — How far is this combination from any explored point in the composition space? Combinations involving rarely studied elements or partner types score higher.
- **Synthesizability** — Can this MXene actually be made? Scored from a curated lookup of experimental synthesis reports (Ti₃C₂: 1.0, Hf₃C₂: 0.2).

### Stage 2.5 — Toxicity Screening
Filters candidates through a biocompatibility assessment based on ISO 10993, EPA IRIS data, and published MXene cytotoxicity studies. Every element in the candidate (metal, carbon/nitrogen, termination, composite partner) is classified into one of four tiers:

| Tier | Classification | Score | Examples |
|------|---------------|-------|----------|
| 1 | Biocompatible | 0.85–1.0 | Ti, Zr, Nb, Ta, C, N, O, Ag, Zn |
| 2 | Low concern | 0.65–0.80 | Hf, Sc, Mo, Sn, Cu, Bi |
| 3 | Moderate concern | 0.40–0.55 | W, F, S, Cl, Se, In |
| 4 | High concern | 0.10–0.20 | Cr, V, Te, Br, Sb, Cd |

The overall biocompatibility score uses a weakest-link principle: the score equals the minimum across all constituent elements. A single toxic element renders the entire composite unsafe regardless of other safe components.

### Stage 3 — Crystal Structure Generation
Generates MXene crystal structures programmatically using the Atomic Simulation Environment (ASE). Structures are built layer by layer from known DFT lattice parameters with appropriate Wyckoff positions, surface termination placement at fcc hollow sites, and vacuum padding for 2D slab calculations. Unknown compositions are estimated via Vegard's law interpolation from Shannon ionic radii.

### Stage 4 — ML-Accelerated Screening
Screens candidate structures using universal machine learning interatomic potentials (CHGNet or MACE):

- **Structural relaxation** via BFGS optimization finds the lowest-energy atomic configuration
- **Formation energy** determines thermodynamic stability relative to elemental ground states
- **Thermoelectric property estimation** uses the Goldsmid-Sharp relation (Seebeck from bandgap), Wiedemann-Franz law (electronic thermal conductivity), and ALIGNN neural network predictions

This replaces months of DFT calculations with minutes of GPU computation while maintaining near-DFT accuracy (~30 meV/atom MAE).

### Stage 5 — Multi-Criteria Ranking
Ranks surviving candidates using TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution), balancing six criteria: predicted ZT, power factor, Seebeck coefficient, thermodynamic stability, synthesizability, and novelty. The weighting scheme prioritizes thermoelectric performance while ensuring candidates are both synthesizable and scientifically novel.

### Stage 6 — DFT Validation
Generates Quantum ESPRESSO input files for the top-ranked candidates, enabling rigorous density functional theory validation of ML predictions. Band structure calculations feed into BoltzTraP2 for Boltzmann transport calculations of Seebeck coefficient, electrical conductivity, and electronic thermal conductivity as functions of temperature and carrier concentration.

## Architecture

```
Semantic Scholar API
        │
        ▼
  Paper Fetcher ──────► papers.jsonl
        │
        ▼
  LLM Data Extractor ──► extracted_records.jsonl
   (Ollama / Qwen2.5)        │
        │                     ▼
        └────────────► SQLite Knowledge Base
                              │
                              ▼
                      Gap Analyzer
                    (composition-space mapping)
                              │
                              ▼
                    Toxicity Screener
                  (biocompatibility filter)
                              │
                              ▼
                   Structure Generator ──► CIF / POSCAR
                         (ASE)                 │
                              │                ▼
                              └──────► ML Stability Screener
                                       (CHGNet / MACE)
                                              │
                                              ▼
                                      TE Property Predictor
                                   (Goldsmid-Sharp / ALIGNN)
                                              │
                                              ▼
                                        TOPSIS Ranker
                                              │
                                              ▼
                                      Top 10–20 candidates
                                              │
                                              ▼
                                    Quantum ESPRESSO (DFT)
                                              │
                                              ▼
                                   Validated TE properties
```

## Design Principles

**Modular and Resumable.** Each stage persists its results to disk and can be run independently. The pipeline tracks its state and resumes from the last completed stage. Checkpointing during long-running stages (LLM extraction, ML screening) enables crash recovery.

**Safety-First Screening.** Toxicity assessment runs *before* computationally expensive ML screening. There is no point spending GPU hours on a material that cannot be worn on skin.

**Conservative by Default.** The toxicity screener uses weakest-link scoring. The TE predictor uses conservative Seebeck estimates (50% of Goldsmid-Sharp maximum). Formation energy thresholds are strict (0.2 eV/atom above hull). It is better to miss a candidate than to recommend a toxic or unstable one.

**Consumer Hardware Compatible.** The entire pipeline runs on a single workstation with a consumer GPU (8 GB VRAM). No HPC cluster, no cloud services, no paid APIs. CHGNet and MACE models fit comfortably in 8 GB VRAM. LLM inference uses quantized models via Ollama.

**Evidence-Backed.** Every prediction carries metadata about its source and confidence level. Literature-extracted values cite the paper ID. ML predictions report the model used and convergence status. The system never fabricates data.

## Key Dependencies

| Component | Role |
|-----------|------|
| [ASE](https://wiki.fysik.dtu.dk/ase/) | Atomic structure manipulation and I/O |
| [pymatgen](https://pymatgen.org/) | Materials analysis and structure operations |
| [CHGNet](https://github.com/CederGroupHub/chgnet) | Universal ML interatomic potential (Berkeley, 2023) |
| [MACE](https://github.com/ACEsuit/mace) | Universal ML interatomic potential (Cambridge, 2022) |
| [ALIGNN](https://github.com/usnistgov/alignn) | Graph neural network property prediction (NIST) |
| [Ollama](https://ollama.com/) | Local LLM inference server |
| [Quantum ESPRESSO](https://www.quantum-espresso.org/) | Density functional theory calculations |
| [BoltzTraP2](https://www.imc.tuwien.ac.at/forschungsbereich_theoretische_chemie/forschungsgruppen/prof_dr_gkh_madsen/boltztrap2/) | Boltzmann transport equation solver |

## Scope and Limitations

MXDiscovery screens **pristine MXene slabs with surface terminations**. It does not currently model:
- MXene–composite interfaces at the atomistic level (the composite partner affects bulk properties but is not included in the crystal structure calculation)
- Mechanical flexibility or strain effects
- Environmental degradation or oxidation kinetics
- Device-level geometry optimization

Thermoelectric property predictions from semi-empirical relations and ML models are screening-quality estimates, not publication-quality values. The top candidates identified by the pipeline require DFT validation (Stage 6) and ultimately experimental synthesis and measurement.

The toxicity database covers elements commonly found in MXenes and their thermoelectric composite partners. It is curated from published biocompatibility data and regulatory classifications but is not exhaustive. For novel element combinations, independent toxicological assessment is recommended before synthesis.

## License

MIT

## Citation

If you use MXDiscovery in your research, please cite:

```
@software{mxdiscovery2026,
  title={MXDiscovery: Computational Pipeline for Non-Toxic MXene Thermoelectric Composite Discovery},
  year={2026},
  url={https://github.com/[username]/MXDiscovery}
}
```
