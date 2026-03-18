"""
pipeline.py - Main MXDiscovery pipeline orchestrating all modules

THIS IS THE ENTRY POINT FOR THE ENTIRE SYSTEM.

PIPELINE STAGES:
    Stage 1: KNOWLEDGE (paper_fetcher → data_extractor → database)
        Input:  Search queries
        Output: Structured SQLite database of MXene TE data from literature

    Stage 2: GAP ANALYSIS (gap_analyzer)
        Input:  Database of known compositions
        Output: Ranked list of unexplored candidates

    Stage 3: STRUCTURE GENERATION (structure_generator)
        Input:  Candidate compositions from gap analysis
        Output: Crystal structure files (CIF/POSCAR)

    Stage 4: ML SCREENING (stability_screener → te_predictor)
        Input:  Crystal structures
        Output: Stability assessment + TE property predictions

    Stage 5: RANKING (ranker)
        Input:  All scores (stability, TE properties, novelty, synthesizability)
        Output: Final ranked list of candidates

    Stage 6: DFT VALIDATION (qe_manager) [for top candidates only]
        Input:  Top 10-20 relaxed structures
        Output: DFT-validated properties

USAGE:
    from src.pipeline import MXDiscoveryPipeline

    pipeline = MXDiscoveryPipeline("config/config.yaml")
    pipeline.run_full()              # Run everything
    pipeline.run_knowledge_stage()   # Run just Stage 1
    pipeline.run_screening_stage()   # Run Stages 2-5
"""

import json
from pathlib import Path
from datetime import datetime

import yaml
from loguru import logger

from .knowledge.paper_fetcher import PaperFetcher
from .knowledge.data_extractor import DataExtractor
from .knowledge.database import MXeneDatabase
from .knowledge.gap_analyzer import GapAnalyzer
from .screening.structure_generator import MXeneStructureGenerator
from .screening.stability_screener import StabilityScreener
from .screening.te_predictor import TEPredictor
from .screening.ranker import CandidateRanker
from .screening.toxicity_screener import ToxicityScreener
from .simulation.qe_manager import QEManager
from .orchestrator.agent import MXDiscoveryAgent


class MXDiscoveryPipeline:
    """
    Main pipeline class that orchestrates the entire discovery workflow.

    DESIGN:
        - Each stage can be run independently (for debugging/iteration)
        - Results from each stage are persisted to disk
        - Pipeline state is tracked so you can resume from any stage
        - All configuration comes from config.yaml

    USAGE EXAMPLES:

        # Full pipeline
        pipeline = MXDiscoveryPipeline()
        pipeline.run_full()

        # Just fetch papers
        pipeline.fetch_papers()

        # Just screen specific candidates
        pipeline.run_screening_stage(top_n=30)

        # Interactive chat mode
        pipeline.chat("What MXene composites have the highest ZT for wearables?")
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")
            self.config = {}

        self.base_dir = Path(self.config.get("project", {}).get("base_dir", "."))

        # Initialize components (lazy - created on first use)
        self._db = None
        self._fetcher = None
        self._extractor = None
        self._gap_analyzer = None
        self._structure_gen = None
        self._screener = None
        self._te_predictor = None
        self._ranker = None
        self._qe_manager = None
        self._toxicity_screener = None
        self._agent = None

        # Pipeline state
        self.state = {
            "papers_fetched": False,
            "data_extracted": False,
            "gap_analyzed": False,
            "structures_generated": False,
            "screening_done": False,
            "te_predicted": False,
            "ranking_done": False,
        }

        logger.info("MXDiscovery Pipeline initialized")

    # ------------------------------------------------------------------
    # Lazy component initialization
    # ------------------------------------------------------------------
    @property
    def db(self) -> MXeneDatabase:
        if self._db is None:
            db_path = self.base_dir / self.config.get("database", {}).get("path", "data/database/mxene_knowledge.db")
            self._db = MXeneDatabase(db_path)
        return self._db

    @property
    def fetcher(self) -> PaperFetcher:
        if self._fetcher is None:
            ss_config = self.config.get("semantic_scholar", {})
            self._fetcher = PaperFetcher(
                output_dir=self.base_dir / "data" / "papers",
                rate_limit=ss_config.get("rate_limit_per_second", 3),
                max_per_query=ss_config.get("max_papers", 500),
            )
        return self._fetcher

    @property
    def extractor(self) -> DataExtractor:
        if self._extractor is None:
            llm_config = self.config.get("llm", {})
            self._extractor = DataExtractor(
                model=llm_config.get("model", "qwen2.5:14b"),
                output_dir=self.base_dir / "data" / "papers",
            )
        return self._extractor

    @property
    def gap_analyzer(self) -> GapAnalyzer:
        if self._gap_analyzer is None:
            self._gap_analyzer = GapAnalyzer(self.db, self.config)
        return self._gap_analyzer

    @property
    def structure_gen(self) -> MXeneStructureGenerator:
        if self._structure_gen is None:
            self._structure_gen = MXeneStructureGenerator(
                output_dir=self.base_dir / "data" / "structures"
            )
        return self._structure_gen

    @property
    def screener(self) -> StabilityScreener:
        if self._screener is None:
            sc = self.config.get("screening", {})
            self._screener = StabilityScreener(
                model=sc.get("ml_potential", "chgnet"),
                energy_cutoff=sc.get("energy_cutoff_ev_per_atom", 0.2),
                force_tol=sc.get("force_tolerance", 0.05),
                max_steps=sc.get("max_relaxation_steps", 500),
                output_dir=self.base_dir / "data" / "structures" / "relaxed",
            )
        return self._screener

    @property
    def te_predictor(self) -> TEPredictor:
        if self._te_predictor is None:
            sc = self.config.get("screening", {})
            self._te_predictor = TEPredictor(
                temperature_k=sc.get("target_temperature_K", 310),
            )
        return self._te_predictor

    @property
    def ranker(self) -> CandidateRanker:
        if self._ranker is None:
            self._ranker = CandidateRanker()
        return self._ranker

    @property
    def toxicity_screener(self) -> ToxicityScreener:
        if not hasattr(self, "_toxicity_screener") or self._toxicity_screener is None:
            self._toxicity_screener = ToxicityScreener()
        return self._toxicity_screener

    @property
    def agent(self) -> MXDiscoveryAgent:
        if self._agent is None:
            llm_config = self.config.get("llm", {})
            self._agent = MXDiscoveryAgent(
                model=llm_config.get("model", "qwen2.5:14b"),
                memory_file=str(self.base_dir / "data" / "database" / "agent_memory.json"),
                db=self.db,
                config=self.config,
            )
        return self._agent

    # ------------------------------------------------------------------
    # Stage 1: Knowledge
    # ------------------------------------------------------------------
    def fetch_papers(self) -> dict:
        """
        STAGE 1a: Fetch MXene thermoelectric papers from Semantic Scholar.

        WHAT HAPPENS:
            1. Sends 10 different search queries to Semantic Scholar API
            2. Collects all matching papers (title, abstract, DOI, etc.)
            3. Deduplicates (same paper can match multiple queries)
            4. Saves to data/papers/papers.jsonl
            5. Also creates papers_summary.csv for quick review
        """
        logger.info("=" * 60)
        logger.info("STAGE 1a: Fetching papers from Semantic Scholar")
        logger.info("=" * 60)

        queries = self.config.get("semantic_scholar", {}).get("search_queries", [
            "MXene thermoelectric",
            "MXene Seebeck coefficient",
        ])

        papers = self.fetcher.fetch_all(queries)
        stats = self.fetcher.get_stats()

        # Load papers into database
        for p in self.fetcher.papers:
            self.db.insert_paper(
                paper_id=p.paper_id,
                title=p.title,
                abstract=p.abstract,
                year=p.year,
                authors=json.dumps(p.authors),
                doi=p.doi,
                venue=p.venue,
                citations=p.citation_count,
                has_pdf=bool(p.open_access_pdf),
                tldr=p.tldr,
            )

        self.state["papers_fetched"] = True
        logger.info(f"Papers fetched: {stats}")
        return stats

    def extract_data(self) -> dict:
        """
        STAGE 1b: Extract structured TE data from paper abstracts using LLM.

        WHAT HAPPENS:
            1. Loads all papers from JSONL file
            2. For each paper with an abstract:
               - Sends abstract to local LLM (Qwen2.5)
               - LLM extracts: composition, properties, synthesis method
               - Response is validated with Pydantic
            3. Saves extracted records to database
            4. Checkpoints every 10 papers (resume-safe)
        """
        logger.info("=" * 60)
        logger.info("STAGE 1b: Extracting structured data with LLM")
        logger.info("=" * 60)

        # Load papers
        papers_data = []
        for p in self.fetcher.papers:
            papers_data.append({
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": p.abstract,
            })

        records = self.extractor.extract_batch(papers_data)

        # Insert into database
        for r in records:
            self.db.insert_te_record(r.model_dump())

        self.state["data_extracted"] = True
        stats = self.extractor.get_stats()
        logger.info(f"Extraction complete: {stats}")
        return stats

    # ------------------------------------------------------------------
    # Stage 2: Gap Analysis
    # ------------------------------------------------------------------
    def run_gap_analysis(self, top_n: int = 50):
        """
        STAGE 2: Identify unexplored MXene composition spaces.

        WHAT HAPPENS:
            1. Enumerates all theoretical MXene + partner combinations
            2. Cross-references with literature database
            3. Scores each unexplored combination by:
               - Novelty (how far from known territory)
               - Analogy (how similar to known good performers)
               - Synthesizability (can it actually be made?)
            4. Returns ranked list of discovery targets
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: Gap Analysis - Finding discovery opportunities")
        logger.info("=" * 60)

        result = self.gap_analyzer.analyze(top_n=top_n)
        self.gap_analyzer.print_report(result)

        self.state["gap_analyzed"] = True
        self._last_gap_result = result
        return result

    # ------------------------------------------------------------------
    # Stage 3-4: Screening
    # ------------------------------------------------------------------
    def run_screening_stage(self, top_n: int = 30):
        """
        STAGES 3-4: Generate structures and screen for stability + TE properties.

        WHAT HAPPENS:
            1. Take top N candidates from gap analysis
            2. Generate crystal structures for each (ASE)
            3. Relax structures with CHGNet/MACE (ML potential)
            4. Compute formation energy → filter stable ones
            5. Predict TE properties for stable candidates
            6. Rank using TOPSIS multi-criteria analysis
        """
        logger.info("=" * 60)
        logger.info("STAGES 3-4: Structure generation + ML screening")
        logger.info("=" * 60)

        # Get candidates from gap analysis
        if not hasattr(self, "_last_gap_result"):
            logger.info("Running gap analysis first...")
            self.run_gap_analysis(top_n=top_n)

        candidates = self._last_gap_result.top_candidates[:top_n]

        # Convert Candidate objects to dicts for structure generator
        cand_dicts = []
        for c in candidates:
            cand_dicts.append({
                "mxene_formula": c.mxene_formula,
                "m_elements": c.m_elements,
                "x_element": c.x_element,
                "stoichiometry": c.stoichiometry,
                "termination": c.termination,
                "composite_partner": c.composite_partner,
                "partner_type": c.partner_type,
                "novelty_score": c.novelty_score,
                "synthesizability": c.synthesizability,
            })

        # Stage 2.5: Toxicity screening (filter BEFORE expensive ML screening)
        logger.info("Running toxicity screening for wearable safety...")
        tox_results = self.toxicity_screener.screen_batch(cand_dicts)
        self.toxicity_screener.print_report(tox_results)

        # Keep only SAFE candidates
        safe_names = set()
        for t in tox_results:
            if t.is_wearable_safe:
                safe_names.add(t.candidate_name)

        cand_dicts_safe = [c for c in cand_dicts
                           if f"{c['mxene_formula']}_{c['termination']}_{c['composite_partner']}" in safe_names
                           or c.get("name", "") in safe_names]

        # If no safe candidates, fall back to CAUTION ones
        if not cand_dicts_safe:
            logger.warning("No SAFE candidates found. Including CAUTION candidates.")
            caution_names = set()
            for t in tox_results:
                if t.classification in ("SAFE", "CAUTION"):
                    caution_names.add(t.candidate_name)
            cand_dicts_safe = [c for c in cand_dicts
                               if f"{c['mxene_formula']}_{c['termination']}_{c['composite_partner']}" in caution_names]

        logger.info(f"After toxicity filter: {len(cand_dicts_safe)}/{len(cand_dicts)} candidates retained")

        # Stage 3: Generate structures
        logger.info(f"Generating {len(cand_dicts_safe)} structures...")
        structures = self.structure_gen.generate_batch(cand_dicts_safe)
        self.state["structures_generated"] = True

        # Stage 4a: ML stability screening
        logger.info("Running ML stability screening...")
        screening_results = self.screener.screen_batch(structures)
        self.state["screening_done"] = True

        # Save screening results to database
        for r in screening_results:
            self.db.insert_screening_result(r)

        # Stage 4b: TE property prediction
        logger.info("Predicting thermoelectric properties...")
        te_results = self.te_predictor.screen_candidates(screening_results)
        self.state["te_predicted"] = True

        # Stage 5: Ranking
        logger.info("Ranking candidates...")
        ranked = self.ranker.rank(
            candidates=screening_results,
            te_predictions=te_results,
            gap_candidates=candidates,
        )
        self.ranker.print_rankings(ranked)
        self.state["ranking_done"] = True

        self._last_ranked = ranked
        return ranked

    # ------------------------------------------------------------------
    # Stage 6: DFT Validation
    # ------------------------------------------------------------------
    def generate_dft_inputs(self, top_n: int = 10):
        """
        STAGE 6: Generate Quantum ESPRESSO input files for top candidates.

        WHAT HAPPENS:
            1. Takes top N ranked candidates
            2. Loads their relaxed structures
            3. Generates QE input files (SCF + bands)
            4. Creates run scripts for WSL2 execution

        NOTE: Actual DFT execution is manual (run in WSL2).
        This just prepares the input files.
        """
        logger.info("=" * 60)
        logger.info("STAGE 6: Generating DFT validation inputs")
        logger.info("=" * 60)

        if not hasattr(self, "_last_ranked"):
            logger.error("Run screening stage first")
            return

        qe = QEManager(
            pseudo_dir=str(self.base_dir / self.config.get("dft", {}).get("pseudopotential_dir", "data/pseudopotentials")),
            output_dir=str(self.base_dir / "data" / "results" / "dft"),
            ecutwfc=self.config.get("dft", {}).get("ecutwfc", 60),
            ecutrho=self.config.get("dft", {}).get("ecutrho", 480),
            kpoints=tuple(self.config.get("dft", {}).get("kpoints", [8, 8, 1])),
        )

        for cand in self._last_ranked[:top_n]:
            relaxed_file = None
            # Find relaxed structure file
            relaxed_dir = self.base_dir / "data" / "structures" / "relaxed"
            possible = list(relaxed_dir.glob(f"*{cand.name}*relaxed.cif"))
            if possible:
                relaxed_file = possible[0]

            if relaxed_file:
                from ase.io import read as ase_read
                atoms = ase_read(str(relaxed_file))
                qe.generate_scf_input(atoms, cand.name, calculation="scf")
                qe.generate_scf_input(atoms, cand.name, calculation="relax")
                qe.generate_bands_input(atoms, cand.name)
                logger.info(f"Generated DFT inputs for {cand.name}")
            else:
                logger.warning(f"No relaxed structure found for {cand.name}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full(self, top_n_screen: int = 30, top_n_dft: int = 10):
        """
        Run the complete discovery pipeline.

        STAGES:
            1a. Fetch papers from Semantic Scholar
            1b. Extract structured data with LLM
            2.  Gap analysis (find unexplored compositions)
            3.  Generate candidate structures
            4.  ML screening (stability + TE properties)
            5.  TOPSIS ranking
            6.  Generate DFT input files for top candidates
        """
        start = datetime.now()
        logger.info("=" * 60)
        logger.info("  MXDiscovery - FULL PIPELINE RUN")
        logger.info(f"  Started: {start}")
        logger.info("=" * 60)

        # Stage 1
        self.fetch_papers()
        self.extract_data()

        # Stage 2
        self.run_gap_analysis(top_n=top_n_screen)

        # Stages 3-5
        self.run_screening_stage(top_n=top_n_screen)

        # Stage 6
        self.generate_dft_inputs(top_n=top_n_dft)

        elapsed = datetime.now() - start
        logger.info(f"\nPipeline complete in {elapsed}")
        logger.info(f"Database: {self.db.get_summary_stats()}")

    # ------------------------------------------------------------------
    # Interactive mode
    # ------------------------------------------------------------------
    def chat(self, message: str) -> str:
        """Send a message to the AI agent."""
        return self.agent.chat(message)

    def interactive(self):
        """Start interactive chat loop."""
        print("\n" + "=" * 60)
        print("  MXDiscovery - Interactive Mode")
        print("  Type 'quit' to exit, 'status' for pipeline state")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "status":
                for k, v in self.state.items():
                    print(f"  {k}: {'Done' if v else 'Pending'}")
                continue

            response = self.chat(user_input)
            print(f"\nMXDiscovery: {response}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    pipeline = MXDiscoveryPipeline()

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "fetch":
            pipeline.fetch_papers()
        elif command == "extract":
            pipeline.extract_data()
        elif command == "gap":
            pipeline.run_gap_analysis()
        elif command == "screen":
            pipeline.run_screening_stage()
        elif command == "full":
            pipeline.run_full()
        elif command == "chat":
            pipeline.interactive()
        else:
            print(f"Unknown command: {command}")
            print("Available: fetch, extract, gap, screen, full, chat")
    else:
        pipeline.interactive()
