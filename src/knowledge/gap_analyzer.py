"""
gap_analyzer.py - Systematic identification of unexplored MXene composition space

THIS IS THE CORE NOVELTY OF THE PROJECT.

PROBLEM:
    MXene research is heavily biased toward Ti3C2Tx (~80% of all papers).
    Most composite partners are PEDOT:PSS, rGO, and CNT.
    The VAST majority of MXene-partner-termination combinations have
    NEVER been studied for thermoelectric properties.

ALGORITHM (Composition Space Mapping):
    1. Define the FULL theoretical composition space:
       - M elements: Ti, V, Nb, Mo, Cr, Zr, Hf, Ta, W, Sc (10)
       - X elements: C, N (2)
       - Stoichiometries: M2X, M3X2, M4X3 (3)
       - Terminations: O, OH, F, Cl, S, Se, Te, Br (8)
       - Composite partners: ~25 categories
       Total theoretical space: 10 × 2 × 3 × 8 × 25 = 12,000 combinations

    2. Map the EXPLORED space from our literature database:
       - Which combinations have published TE data?
       - Which have any published data (even non-TE)?

    3. SCORE unexplored combinations by discovery potential:
       - Analogical reasoning: if Ti3C2Tx/PEDOT:PSS works well,
         V2CTx/PEDOT:PSS might too (same partner, similar MXene)
       - Property interpolation: estimate properties based on
         neighboring compositions that have data
       - Novelty score: combinations far from any explored point
         are more novel but also more risky
       - Synthesizability: some MXenes are known to be synthesizable,
         others are only theoretical

    4. RANK candidates for screening:
       - High discovery potential + reasonable synthesizability = TOP PRIORITY
       - Novel but hard to make = COMPUTATIONAL ONLY
       - Already well-studied = SKIP

DATA STRUCTURES:
    - CompositionSpace: Full grid of all possible combinations
    - ExplorationMap: Boolean matrix (explored/unexplored)
    - CandidateScore: Weighted score for each unexplored combination
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Optional

import numpy as np
from loguru import logger

from .database import MXeneDatabase


@dataclass
class Candidate:
    """A single unexplored MXene composite candidate."""
    mxene_formula: str          # e.g., "Ti3C2"
    m_elements: list[str]       # e.g., ["Ti"]
    x_element: str              # "C" or "N"
    stoichiometry: str          # "M2X", "M3X2", "M4X3"
    termination: str            # "O", "OH", etc.
    composite_partner: str      # "PEDOT:PSS", "Bi2Te3", etc.
    partner_type: str           # polymer, carbon, chalcogenide, etc.

    # Scores (computed by analyze())
    novelty_score: float = 0.0        # How far from explored space
    analogy_score: float = 0.0        # How similar to known good performers
    synthesizability: float = 0.0     # How likely this can be made
    overall_score: float = 0.0        # Weighted combination

    # Estimated properties (from interpolation)
    estimated_seebeck: Optional[float] = None
    estimated_pf: Optional[float] = None


@dataclass
class GapAnalysisResult:
    """Complete results of a gap analysis run."""
    total_theoretical: int
    total_explored: int
    total_unexplored: int
    exploration_rate: float
    top_candidates: list[Candidate]
    underexplored_mxenes: list[str]     # MXenes with < 3 partner studies
    underexplored_partners: list[str]   # Partners studied with < 3 MXenes
    hot_spots: list[dict]               # Well-studied combinations (for validation)


class GapAnalyzer:
    """
    Analyzes the MXene composition space to find discovery opportunities.

    HOW IT WORKS:

    Step 1 - Build Composition Space:
        Generate every valid MXene formula + termination + partner combination.
        Uses combinatorial enumeration (itertools.product).

    Step 2 - Map Explored Territory:
        Query the database to find which combinations have published data.
        Build a binary matrix: 1 = explored, 0 = unexplored.

    Step 3 - Score Candidates:
        For each unexplored combination, compute three scores:

        a) NOVELTY SCORE (0-1):
           How "far" is this combination from any explored combination?
           Uses a simple distance metric:
           - Same M element family? -0.3 (less novel)
           - Same partner type? -0.2 (less novel)
           - Same stoichiometry? -0.1 (less novel)
           Higher = more novel = potentially higher impact publication

        b) ANALOGY SCORE (0-1):
           How similar is this to a KNOWN GOOD thermoelectric?
           If Ti3C2Tx/PEDOT:PSS has high PF, then:
           - V2CTx/PEDOT:PSS gets high analogy score (same partner)
           - Ti3C2Tx/PANI gets high analogy score (same MXene)
           Higher = more likely to have good TE properties

        c) SYNTHESIZABILITY SCORE (0-1):
           Can this MXene actually be made?
           Based on known experimental synthesis reports:
           - Ti3C2Tx: 1.0 (routinely made)
           - Mo2TiC2Tx: 0.8 (made by several groups)
           - Hf2C: 0.2 (theoretical only)

    Step 4 - Rank and Filter:
        overall_score = w1*analogy + w2*novelty + w3*synthesizability
        Default weights: analogy=0.4, novelty=0.3, synthesizability=0.3
        (We want candidates that are novel AND likely to work AND can be made)

    RETURNS:
        Ranked list of candidates sorted by overall_score.
    """

    # Known synthesizability of MXene base compositions (from literature)
    # 1.0 = routinely synthesized, 0.0 = never made
    SYNTHESIZABILITY = {
        "Ti2C": 0.9, "Ti3C2": 1.0, "Ti4C3": 0.7,
        "Ti2N": 0.6, "Ti3N2": 0.4, "Ti4N3": 0.3,
        "V2C": 0.8, "V2N": 0.5,
        "Nb2C": 0.7, "Nb4C3": 0.5,
        "Mo2C": 0.6, "Mo2TiC2": 0.7, "Mo2Ti2C3": 0.6,
        "Cr2TiC2": 0.5,
        "Ta2C": 0.4,
        "Zr3C2": 0.3,
        "Hf3C2": 0.2,
        "W2C": 0.3,
        "Sc2C": 0.2,
    }

    def __init__(self, db: MXeneDatabase, config: dict = None):
        self.db = db
        self.config = config or {}

        # Load from config or use defaults
        mat = self.config.get("materials", {})
        self.m_elements = mat.get("m_elements", ["Ti", "V", "Nb", "Mo", "Cr", "Zr", "Hf", "Ta", "W", "Sc"])
        self.x_elements = mat.get("x_elements", ["C", "N"])
        self.terminations = mat.get("terminations", ["O", "OH", "F", "Cl", "S", "Se", "Te", "Br"])
        self.stoichiometries = [
            ("M2X", 2, 1),    # M_{n+1}X_n where n=1
            ("M3X2", 3, 2),   # n=2
            ("M4X3", 4, 3),   # n=3
        ]

        partners = mat.get("te_composite_partners", {})
        self.partners = {}
        for ptype, plist in partners.items():
            for p in plist:
                self.partners[p] = ptype

        if not self.partners:
            # Defaults if not in config
            self.partners = {
                "PEDOT:PSS": "polymer", "PANI": "polymer", "PPy": "polymer",
                "SWCNT": "carbon", "MWCNT": "carbon", "rGO": "carbon",
                "Bi2Te3": "chalcogenide", "SnSe": "chalcogenide",
                "ZnO": "oxide", "Ag_NW": "metal",
            }

    def _generate_mxene_formulas(self) -> list[tuple[str, list[str], str, str]]:
        """
        Generate all valid MXene formulas.

        RETURNS: list of (formula, m_elements, x_element, stoich_label)
        Example: ("Ti3C2", ["Ti"], "C", "M3X2")
        """
        formulas = []
        for m_el in self.m_elements:
            for x_el in self.x_elements:
                for label, m_count, x_count in self.stoichiometries:
                    formula = f"{m_el}{m_count}{x_el}{x_count}" if x_count > 1 else f"{m_el}{m_count}{x_el}"
                    formulas.append((formula, [m_el], x_el, label))
        return formulas

    def analyze(
        self,
        weight_analogy: float = 0.4,
        weight_novelty: float = 0.3,
        weight_synth: float = 0.3,
        top_n: int = 50,
    ) -> GapAnalysisResult:
        """
        Run full gap analysis and return ranked candidates.

        ALGORITHM:
            1. Generate all theoretical combinations
            2. Load explored combinations from database
            3. Score each unexplored combination
            4. Rank by overall score
            5. Identify underexplored regions
        """
        # Step 1: Generate full composition space
        formulas = self._generate_mxene_formulas()
        all_candidates = []

        for formula, m_els, x_el, stoich in formulas:
            for term in self.terminations:
                for partner, ptype in self.partners.items():
                    all_candidates.append(Candidate(
                        mxene_formula=formula,
                        m_elements=m_els,
                        x_element=x_el,
                        stoichiometry=stoich,
                        termination=term,
                        composite_partner=partner,
                        partner_type=ptype,
                    ))

        total_theoretical = len(all_candidates)
        logger.info(f"Total theoretical composition space: {total_theoretical}")

        # Step 2: Load explored combinations
        gap_data = self.db.gap_analysis()
        explored_set = set(gap_data["explored"])
        top_performers = self.db.query_top_performers("power_factor", limit=50)

        # Build lookup: which (composition, partner) pairs have good TE performance?
        good_compositions = set()
        good_partners = set()
        for rec in top_performers:
            if rec.get("mxene_composition"):
                good_compositions.add(rec["mxene_composition"])
            if rec.get("composite_partner"):
                good_partners.add(rec["composite_partner"])

        # Step 3: Filter to unexplored and score
        unexplored = []
        for cand in all_candidates:
            # Check if explored (normalize: add Tx suffix for matching)
            comp_key = cand.mxene_formula + "Tx"
            if (comp_key, cand.composite_partner) in explored_set:
                continue
            if (cand.mxene_formula, cand.composite_partner) in explored_set:
                continue

            # --- Novelty Score ---
            # Higher if this M element and partner type are rarely studied together
            m_el_explored = sum(1 for e in explored_set if cand.m_elements[0] in e[0])
            ptype_explored = sum(
                1 for e in explored_set
                if self.partners.get(e[1]) == cand.partner_type
            )
            total_explored = len(explored_set) or 1
            m_rarity = 1.0 - min(m_el_explored / total_explored, 1.0)
            p_rarity = 1.0 - min(ptype_explored / total_explored, 1.0)
            cand.novelty_score = 0.6 * m_rarity + 0.4 * p_rarity

            # --- Analogy Score ---
            # Higher if similar compositions or partners are known to perform well
            comp_sim = 1.0 if any(cand.m_elements[0] in gc for gc in good_compositions) else 0.0
            partner_sim = 1.0 if cand.composite_partner in good_partners else 0.5
            cand.analogy_score = 0.5 * comp_sim + 0.5 * partner_sim

            # --- Synthesizability Score ---
            base = cand.mxene_formula.rstrip("0123456789")
            # Try exact match first, then partial
            cand.synthesizability = self.SYNTHESIZABILITY.get(
                cand.mxene_formula,
                self.SYNTHESIZABILITY.get(base, 0.3)
            )

            # --- Overall Score ---
            cand.overall_score = (
                weight_analogy * cand.analogy_score
                + weight_novelty * cand.novelty_score
                + weight_synth * cand.synthesizability
            )

            unexplored.append(cand)

        # Step 4: Rank
        unexplored.sort(key=lambda c: c.overall_score, reverse=True)

        # Step 5: Identify underexplored regions
        comp_coverage = gap_data.get("composition_coverage", {})
        partner_coverage = gap_data.get("partner_coverage", {})

        underexplored_mxenes = [c for c, n in comp_coverage.items() if n < 3]
        underexplored_partners = [p for p, n in partner_coverage.items() if n < 3]

        result = GapAnalysisResult(
            total_theoretical=total_theoretical,
            total_explored=len(explored_set),
            total_unexplored=len(unexplored),
            exploration_rate=len(explored_set) / max(total_theoretical, 1) * 100,
            top_candidates=unexplored[:top_n],
            underexplored_mxenes=underexplored_mxenes,
            underexplored_partners=underexplored_partners,
            hot_spots=[dict(r) for r in top_performers[:10]],
        )

        logger.info(
            f"Gap analysis complete: {result.total_explored}/{result.total_theoretical} "
            f"explored ({result.exploration_rate:.1f}%), "
            f"{len(result.top_candidates)} top candidates identified"
        )
        return result

    def print_report(self, result: GapAnalysisResult):
        """Print human-readable gap analysis report."""
        print("\n" + "=" * 70)
        print("  MXene THERMOELECTRIC COMPOSITION SPACE - GAP ANALYSIS")
        print("=" * 70)
        print(f"  Total theoretical combinations: {result.total_theoretical:,}")
        print(f"  Explored in literature:         {result.total_explored:,}")
        print(f"  UNEXPLORED:                     {result.total_unexplored:,}")
        print(f"  Exploration rate:               {result.exploration_rate:.1f}%")
        print()
        print("  TOP 20 DISCOVERY CANDIDATES:")
        print("  " + "-" * 66)
        print(f"  {'Rank':<5} {'MXene':<12} {'Term':<5} {'Partner':<15} {'Score':<7} {'Synth':<6}")
        print("  " + "-" * 66)
        for i, c in enumerate(result.top_candidates[:20], 1):
            print(
                f"  {i:<5} {c.mxene_formula+'Tx':<12} {c.termination:<5} "
                f"{c.composite_partner:<15} {c.overall_score:.3f}  {c.synthesizability:.1f}"
            )
        print()
        if result.underexplored_mxenes:
            print(f"  Underexplored MXenes (< 3 partners tested):")
            print(f"    {', '.join(result.underexplored_mxenes[:10])}")
        if result.underexplored_partners:
            print(f"  Underexplored partners (< 3 MXenes tested):")
            print(f"    {', '.join(result.underexplored_partners[:10])}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_dir = Path(config["project"]["base_dir"])
    db_path = base_dir / config["database"]["path"]

    db = MXeneDatabase(db_path)
    analyzer = GapAnalyzer(db, config)
    result = analyzer.analyze(top_n=50)
    analyzer.print_report(result)

    # Save top candidates to JSON for Stage 3 (structure generation)
    candidates_path = base_dir / "data" / "gap_analysis_candidates.json"
    candidates_data = []
    for c in result.top_candidates:
        candidates_data.append({
            "mxene_formula": c.mxene_formula,
            "m_elements": c.m_elements,
            "x_element": c.x_element,
            "stoichiometry": c.stoichiometry,
            "termination": c.termination,
            "composite_partner": c.composite_partner,
            "partner_type": c.partner_type,
            "overall_score": c.overall_score,
            "novelty_score": c.novelty_score,
            "analogy_score": c.analogy_score,
            "synthesizability": c.synthesizability,
        })
    with open(candidates_path, "w") as f:
        json.dump(candidates_data, f, indent=2)
    print(f"\nSaved {len(candidates_data)} candidates to {candidates_path}")

    db.close()
