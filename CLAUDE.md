# MXDiscovery - Project Intelligence

## What Is This Project?

MXDiscovery is a computational pipeline for discovering novel non-toxic MXene composites for wearable thermoelectric energy harvesting. It is NOT a chatbot — it is a scientific discovery engine with a chat interface.

## Who Is The User?

- **Name:** SriNikhil
- **Position:** Masters student at Shizuoka University, Japan
- **Lab:** Ikeda-Hamasaki laboratory, Advanced Device Research Division, Research institute of electronics, Shizuoka University
- **Hardware:** Intel i7, 8 GB RAM, NVIDIA RTX 4060 8 GB VRAM, Windows 11
- **Python:** 3.11.9 (venv at D:/MXDiscovery/venv/)
- **Budget:** Zero — open-source only
- **Background:** Materials scientist with Python experience, NOT a CS/AI person
- **Career goal:** PhD admission (using this work as portfolio) + build lab reputation in materials informatics

## Project Goals (Priority Order)

1. **Discover a new non-toxic wearable MXene composite** — computationally predict AND experimentally validate in their lab
2. **Contribute to materials informatics as a field** — not just use tools, but define methodology
3. **Publish in top journals** — Paper 1: Digital Discovery (RSC), Paper 2: npj Computational Materials
4. **Collaborate with top universities** — Priority: USA (Gogotsi/Drexel, Anasori/Purdue, Ong/UCSD) > Japan (NIMS) > Europe (Cambridge) > India (IIT/IISc)

## Application Focus

- **Wearable thermoelectrics** — body temperature ~310K
- **Non-toxic constraint** — safe for skin contact (ISO 10993 compliance)
- **Safe M-elements:** Ti, Zr, Nb, Ta, Hf, Sc, Mo
- **Unsafe M-elements:** V, Cr, W (excluded from screening)
- **Safe terminations:** O, OH only (F, Cl, Se, Te, Br excluded)
- **Target properties:** Seebeck coefficient, electrical conductivity, thermal conductivity, power factor (PF = S²σ), ZT = PF×T/κ

## Architecture

```
Stage 1a: paper_fetcher.py    → Semantic Scholar API → papers.jsonl
Stage 1b: data_extractor.py   → Ollama LLM extraction → extracted_records.jsonl
Stage 1c: database.py         → SQLite knowledge base
Stage 2:  gap_analyzer.py     → Composition space gap analysis → ranked candidates
Stage 2.5: toxicity_screener.py → Filter unsafe candidates for wearable use
Stage 3:  structure_generator.py → ASE crystal structures → CIF/POSCAR
Stage 4a: stability_screener.py → CHGNet/MACE ML screening → formation energy
Stage 4b: te_predictor.py     → TE property prediction (Goldsmid-Sharp, ALIGNN)
Stage 5:  ranker.py           → TOPSIS multi-criteria ranking
Stage 6:  qe_manager.py       → Quantum ESPRESSO DFT validation (WSL2)
Agent:    agent.py             → LLM orchestrator (Qwen2.5 via Ollama)
Entry:    pipeline.py          → Main orchestrator connecting all stages
```

## Key Technical Decisions

- **Python 3.11** (not 3.13) — PyTorch requires ≤3.12
- **BoltzTraP2** — NOT installed on Windows (needs Fortran). Will install in WSL2 for Stage 6 only.
- **CHGNet** as primary ML potential (MACE as optional alternative)
- **Qwen2.5:7b** via Ollama (NOT 14b — 14b crashed with 8GB RAM). 7b fits entirely in GPU VRAM with `num_gpu: 99`, leaving RAM free for Python/ASE. 7b is sufficient because LLM only does text extraction and routing, not scientific reasoning.
- **SQLite** for structured data (not Postgres — zero config)
- **Toxicity screening uses weakest-link scoring** — overall score = minimum element score
- **TOPSIS** for multi-criteria ranking (weights: ZT=0.25, PF=0.20, S=0.15, stability=0.15, synth=0.15, novelty=0.10)

## Novel Contributions (What Makes This Publishable)

1. **Composition-space gap analysis** for MXene thermoelectrics — systematic identification of unexplored regions
2. **Toxicity-constrained screening** — biocompatibility filter integrated into computational discovery (new for any 2D material)
3. **CHGNet/MACE benchmark for MXene interfaces** — first reported accuracy data (ZERO prior papers)
4. **Body-temperature TE screening at 310K** — most TE screening targets >500K
5. **Fully open-source, consumer GPU** — democratized materials discovery
6. **TOPSIS for computational discovery** — never used in materials screening pipelines before
7. **LLM extraction with local 7B model** — prior work used GPT-4 cloud; nobody used local open-source LLM

## Publication Strategy (11 Possible Papers)

Priority order for PhD admission + lab reputation:

| # | Paper | Target Journal | Type | Timeline |
|---|-------|---------------|------|----------|
| 1 | CHGNet/MACE benchmark on MXenes | npj Computational Materials | Benchmark | Month 2-3 |
| 2 | MXDiscovery pipeline paper | Digital Discovery (RSC) | Methodology | Month 3-5 |
| 3 | MXeneTE structured database | Scientific Data (Nature) | Data resource | Month 4-5 |
| 4 | ML-predicted stable MXene TE composites | npj Computational Materials | Discovery | Month 5-7 |
| 5 | Experimental validation of predictions | Nature Communications | Discovery+Exp | Month 6-9 |
| 6 | Materials informatics for 2D materials review | Materials Today | Review | Month 7-8 |
| 7 | Composition-space gap analysis framework | Chemistry of Materials | Methodology | Split from #2 |
| 8 | Toxicity-constrained materials screening | Matter / ACS Sustainable Chem | Methodology | Split from #2 |
| 9 | LLM extraction accuracy evaluation | J. Chem. Inf. Model. | Benchmark | Split from #2 |
| 10 | Democratized discovery on consumer GPU | Digital Discovery | Perspective | Anytime |
| 11 | Materials informatics curriculum | J. Materials Education | Education | Anytime |

Key literature gaps confirmed by web search (March 2026):
- CHGNet/MACE on MXenes: **ZERO papers** exist
- Composition-space gap analysis for MXenes: **ZERO papers**
- Toxicity in computational materials discovery: **ONE paper** (MOFs only, Matter Jan 2025)
- TOPSIS in computational screening: **ZERO papers** (only used for commercial material selection)
- Local LLM for materials extraction: **ZERO papers** (all use GPT-4 cloud)
- Full pipeline on consumer GPU: **ZERO papers**

## Documentation Requirements

- **EVERY module** must have detailed docstrings explaining: algorithm, data structures, techniques, and scientific rationale
- **Implementation guide** lives at: docs/MXDiscovery_Implementation_Guide.docx
- User wants to UNDERSTAND everything, not just run it — explain the "why" not just the "how"
- Bridge materials science and CS concepts (user knows materials, learning CS)

## File Locations

- Project root: `D:/MXDiscovery/`
- Config: `D:/MXDiscovery/config/config.yaml`
- Source code: `D:/MXDiscovery/src/`
- Data: `D:/MXDiscovery/data/` (papers, structures, results, database)
- Docs: `D:/MXDiscovery/docs/`
- Virtual env: `D:/MXDiscovery/venv/` (Python 3.11.9)

## Current State (Update This After Each Session)

- [x] Project structure created
- [x] All core modules written (paper_fetcher, data_extractor, database, gap_analyzer, structure_generator, stability_screener, te_predictor, ranker, toxicity_screener, qe_manager, agent, pipeline)
- [x] Virtual environment created (Python 3.11)
- [x] Dependencies installed (PyTorch CUDA + requirements.txt)
- [x] Toxicity screener tested and working
- [x] Implementation guide document generated
- [x] Ollama installed and qwen2.5:7b pulled (7b not 14b — 14b crashes on 8GB RAM)
- [x] LLM tested and working (JSON extraction confirmed)
- [x] GitHub repo created and committed
- [x] README.md written (explanatory, no personal goals, no usage instructions)
- [x] MIT License selected
- [x] Stage 1a: Paper fetching — 2000 papers from OpenAlex (Semantic Scholar IP-blocked; OpenAlex fallback works perfectly)
- [x] Stage 1b: LLM extraction — 123/123 TE-relevant papers extracted (18 compositions, 42 partners)
- [x] Stage 1→DB: Loaded into SQLite (papers + te_records tables)
- [x] Stage 2: Gap analysis — 11,040 theoretical space, only 26 explored (0.2%), 50 top candidates identified
- [x] Toxicity screening — 20 safe candidates for wearable use (Mo2C, Ti2C, Mo-nitrides with various partners)
- [x] Stage 3: Structure generation — 20 POSCAR files generated (ASE, all stoichiometries)
- [x] Stage 4: CHGNet stability screening — ALL 20 candidates thermodynamically stable (23.7s on GPU)
  - Ti2CO2: E_f = -1.749 eV/atom (most stable)
  - Mo2NO2: E_f = -0.906 eV/atom
  - Mo2CO2: E_f = -0.572 eV/atom
- [ ] Stage 5: TE property prediction (Goldsmid-Sharp, ALIGNN, TOPSIS ranking)
- [ ] Stage 6: DFT validation (Quantum ESPRESSO via WSL2)
- [ ] Collaboration emails sent
- [ ] Paper 1 draft (Digital Discovery)
- [ ] Paper 2 draft (npj Computational Materials)

## Key Discovery Results

- **Mo2CTx** is the most underexplored MXene for thermoelectrics — appears in 0 published TE studies
- **Ti2CTx** with any composite partner has NO published TE data (despite Ti2C being easy to synthesize)
- **Mo-nitrides (Mo2N, Mo3N2)** are extremely stable but completely unstudied for TE
- Best partner categories: conducting polymers (PEDOT:PSS) and carbon materials (SWCNT, rGO, graphene)
- Safe M-elements confirmed: Ti, Zr, Nb, Ta, Hf, Sc, Mo (V and Cr excluded by toxicity screener)

## Rules

1. **ALWAYS commit to GitHub after implementing a new feature, change, or fix.** Use descriptive commit messages explaining what was added/changed and why. Tag significant milestones (v0.1.0, v0.2.0, etc.).
2. **NEVER fabricate scientific data** — if unsure, say so explicitly
3. **ALWAYS explain algorithms and techniques** in docstrings and to the user — they want to learn
4. **ALWAYS run toxicity screening** before ML screening — don't waste GPU time on unsafe candidates
5. **ALWAYS validate against literature** — predictions must be checkable
6. **Keep the pipeline modular** — each stage independently runnable and testable
7. **Prefer conservative estimates** — better to miss a candidate than to recommend a toxic one
8. **Update this CLAUDE.md** whenever project state or goals change significantly
