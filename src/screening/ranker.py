"""
ranker.py - Multi-criteria ranking of MXene thermoelectric candidates

ALGORITHM:
    After screening and TE property prediction, we have multiple metrics
    per candidate (stability, Seebeck, conductivity, ZT, novelty, etc.).
    We need to combine these into a single ranking.

    METHOD: TOPSIS (Technique for Order of Preference by Similarity to
            Ideal Solution)

    TOPSIS works by:
    1. Build a decision matrix (rows = candidates, columns = criteria)
    2. Normalize each column (0-1 scale)
    3. Apply weights to each criterion
    4. Find the "ideal best" and "ideal worst" solutions
    5. Compute distance of each candidate to ideal best and ideal worst
    6. Rank by closeness to ideal solution

    WHY TOPSIS:
    - Handles multiple criteria with different units
    - Accounts for trade-offs (high Seebeck often means low conductivity)
    - Widely used in materials selection literature
    - Simple to implement and explain

    CRITERIA AND WEIGHTS (for wearable thermoelectrics):
        - ZT (or estimated ZT):        weight = 0.25 (maximize)
        - Power factor:                 weight = 0.20 (maximize)
        - Seebeck coefficient:          weight = 0.15 (maximize)
        - Stability (formation energy): weight = 0.15 (minimize = more negative is better)
        - Synthesizability:             weight = 0.15 (maximize)
        - Novelty:                      weight = 0.10 (maximize)

    Weights favor ZT and PF because those directly determine device performance.
    Synthesizability is important because a prediction is useless if you can't make it.
    Novelty gets lower weight because we want GOOD materials, not just NEW ones.

DATA STRUCTURES:
    - RankedCandidate: combines all scores into one object
    - RankingResult: full ranked list with metadata
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class RankedCandidate:
    """A fully ranked MXene composite candidate."""
    rank: int = 0
    name: str = ""
    mxene_formula: str = ""
    termination: str = ""
    composite_partner: str = ""
    partner_type: str = ""

    # Properties
    formation_energy: float = 0.0
    is_stable: bool = False
    seebeck: Optional[float] = None
    conductivity: Optional[float] = None
    thermal_cond: Optional[float] = None
    power_factor: Optional[float] = None
    zt: Optional[float] = None

    # Scores
    novelty_score: float = 0.0
    synthesizability: float = 0.0
    topsis_score: float = 0.0

    # Metadata
    method: str = ""
    confidence: str = "low"
    recommendation: str = ""  # "high priority", "moderate", "low priority"


class CandidateRanker:
    """
    Ranks MXene TE candidates using multi-criteria decision analysis (TOPSIS).

    HOW IT WORKS:

    INPUT: Combined data from:
        - gap_analyzer.py (novelty_score, synthesizability)
        - stability_screener.py (formation_energy, is_stable)
        - te_predictor.py (Seebeck, conductivity, PF, ZT)

    STEP 1 - Build Decision Matrix:
        Each row = one candidate
        Each column = one criterion
        Values are the raw property values

        Example:
        Candidate          ZT    PF     S      E_f    Synth  Novel
        Ti3C2O2/PEDOT     0.15  120   45.0   -0.8    1.0    0.3
        V2CO2/Bi2Te3      0.28   85   62.0   -0.5    0.8    0.7
        Mo2CS2/SWCNT      0.05  200   30.0   -0.3    0.6    0.9

    STEP 2 - Normalize:
        Each column is normalized using vector normalization:
        r_ij = x_ij / sqrt(sum(x_ij²))
        This puts all criteria on the same scale regardless of units.

    STEP 3 - Weight:
        Multiply each normalized column by its weight.
        v_ij = w_j × r_ij

    STEP 4 - Ideal Solutions:
        Ideal best: max of each "maximize" criterion, min of "minimize"
        Ideal worst: opposite

    STEP 5 - Distances:
        D+ = Euclidean distance to ideal best
        D- = Euclidean distance to ideal worst

    STEP 6 - Closeness:
        C = D- / (D+ + D-)
        Higher C = better candidate
        C=1 means the candidate IS the ideal solution
    """

    DEFAULT_WEIGHTS = {
        "zt": 0.25,
        "power_factor": 0.20,
        "seebeck": 0.15,
        "stability": 0.15,
        "synthesizability": 0.15,
        "novelty": 0.10,
    }

    # Whether to maximize (+1) or minimize (-1) each criterion
    DIRECTIONS = {
        "zt": 1,
        "power_factor": 1,
        "seebeck": 1,
        "stability": -1,  # more negative formation energy = better
        "synthesizability": 1,
        "novelty": 1,
    }

    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def rank(
        self,
        candidates: list[dict],
        te_predictions: list = None,
        gap_candidates: list = None,
    ) -> list[RankedCandidate]:
        """
        Rank candidates using TOPSIS.

        PARAMS:
            candidates: list of dicts from stability_screener (with formation_energy, is_stable)
            te_predictions: list of TEProperties from te_predictor
            gap_candidates: list of Candidate from gap_analyzer (with novelty, synthesizability)
        """
        if not candidates:
            return []

        # Merge data sources
        merged = self._merge_data(candidates, te_predictions, gap_candidates)

        if len(merged) < 2:
            # TOPSIS needs at least 2 candidates
            return [self._to_ranked(m, rank=1) for m in merged]

        # Build decision matrix
        criteria = list(self.weights.keys())
        matrix = np.zeros((len(merged), len(criteria)))

        for i, cand in enumerate(merged):
            for j, crit in enumerate(criteria):
                matrix[i, j] = cand.get(crit, 0.0) or 0.0

        # TOPSIS
        scores = self._topsis(matrix, criteria)

        # Sort by TOPSIS score
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(ranked_indices, 1):
            cand_data = merged[idx]
            cand_data["topsis_score"] = scores[idx]
            rc = self._to_ranked(cand_data, rank=rank)
            results.append(rc)

        return results

    def _topsis(self, matrix: np.ndarray, criteria: list[str]) -> np.ndarray:
        """
        Run TOPSIS algorithm on decision matrix.

        RETURNS: array of closeness scores (0 to 1) for each candidate
        """
        n_candidates, n_criteria = matrix.shape

        # Step 2: Vector normalization
        # For each column, divide by the L2 norm
        norms = np.sqrt(np.sum(matrix ** 2, axis=0))
        norms[norms == 0] = 1  # avoid division by zero
        normalized = matrix / norms

        # Step 3: Apply weights
        weight_array = np.array([self.weights[c] for c in criteria])
        weighted = normalized * weight_array

        # Step 4: Ideal solutions
        ideal_best = np.zeros(n_criteria)
        ideal_worst = np.zeros(n_criteria)

        for j, crit in enumerate(criteria):
            direction = self.DIRECTIONS.get(crit, 1)
            if direction == 1:  # maximize
                ideal_best[j] = np.max(weighted[:, j])
                ideal_worst[j] = np.min(weighted[:, j])
            else:  # minimize
                ideal_best[j] = np.min(weighted[:, j])
                ideal_worst[j] = np.max(weighted[:, j])

        # Step 5: Distances
        d_best = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
        d_worst = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))

        # Step 6: Closeness coefficient
        denominator = d_best + d_worst
        denominator[denominator == 0] = 1
        closeness = d_worst / denominator

        return closeness

    def _merge_data(self, candidates, te_predictions, gap_candidates):
        """Merge data from different pipeline stages into unified records."""
        merged = []

        # Index TE predictions by name
        te_by_name = {}
        if te_predictions:
            for tp in te_predictions:
                te_by_name[tp.name] = tp

        # Index gap candidates by formula+partner
        gap_by_key = {}
        if gap_candidates:
            for gc in gap_candidates:
                key = f"{gc.mxene_formula}_{gc.termination}_{gc.composite_partner}"
                gap_by_key[key] = gc

        for cand in candidates:
            name = cand.get("name", "")
            record = {
                "name": name,
                "mxene_formula": cand.get("mxene_formula", ""),
                "termination": cand.get("termination", ""),
                "composite_partner": cand.get("composite_partner", ""),
                "formation_energy": cand.get("formation_energy", 0),
                "is_stable": cand.get("is_stable", False),
                "stability": -(cand.get("formation_energy", 0) or 0),  # negate: more stable = higher
            }

            # Merge TE predictions
            te = te_by_name.get(name)
            if te:
                record["seebeck"] = te.seebeck_coefficient or 0
                record["power_factor"] = te.power_factor or 0
                record["zt"] = te.zt_value or 0
                record["conductivity"] = te.electrical_conductivity or 0
                record["thermal_cond"] = te.thermal_conductivity or 0
                record["method"] = te.method
                record["confidence"] = te.confidence

            # Merge gap analysis scores
            key = f"{record['mxene_formula']}_{record['termination']}_{record['composite_partner']}"
            gap = gap_by_key.get(key)
            if gap:
                record["novelty"] = gap.novelty_score
                record["synthesizability"] = gap.synthesizability
            else:
                record["novelty"] = 0.5
                record["synthesizability"] = 0.5

            merged.append(record)

        return merged

    def _to_ranked(self, data: dict, rank: int) -> RankedCandidate:
        """Convert merged dict to RankedCandidate."""
        topsis = data.get("topsis_score", 0)

        if topsis > 0.7:
            recommendation = "HIGH PRIORITY - strong candidate for experimental validation"
        elif topsis > 0.4:
            recommendation = "MODERATE - worth further computational investigation"
        else:
            recommendation = "LOW PRIORITY - unlikely to outperform known materials"

        return RankedCandidate(
            rank=rank,
            name=data.get("name", ""),
            mxene_formula=data.get("mxene_formula", ""),
            termination=data.get("termination", ""),
            composite_partner=data.get("composite_partner", ""),
            formation_energy=data.get("formation_energy", 0),
            is_stable=data.get("is_stable", False),
            seebeck=data.get("seebeck"),
            conductivity=data.get("conductivity"),
            thermal_cond=data.get("thermal_cond"),
            power_factor=data.get("power_factor"),
            zt=data.get("zt"),
            novelty_score=data.get("novelty", 0),
            synthesizability=data.get("synthesizability", 0),
            topsis_score=topsis,
            method=data.get("method", ""),
            confidence=data.get("confidence", "low"),
            recommendation=recommendation,
        )

    def print_rankings(self, ranked: list[RankedCandidate], top_n: int = 20):
        """Print formatted ranking table."""
        print("\n" + "=" * 90)
        print("  MXENE THERMOELECTRIC CANDIDATE RANKINGS (TOPSIS)")
        print("=" * 90)
        print(f"  {'#':<4} {'MXene':<10} {'Tx':<4} {'Partner':<12} "
              f"{'ZT':<7} {'PF':<8} {'S(µV/K)':<8} {'E_f':<7} {'Score':<7} {'Priority'}")
        print("  " + "-" * 86)

        for c in ranked[:top_n]:
            zt_str = f"{c.zt:.3f}" if c.zt else "  -  "
            pf_str = f"{c.power_factor:.1f}" if c.power_factor else "  -  "
            s_str = f"{c.seebeck:.1f}" if c.seebeck else "  -  "
            ef_str = f"{c.formation_energy:.3f}"

            priority = "HIGH" if c.topsis_score > 0.7 else "MED" if c.topsis_score > 0.4 else "LOW"

            print(f"  {c.rank:<4} {c.mxene_formula:<10} {c.termination:<4} "
                  f"{c.composite_partner:<12} {zt_str:<7} {pf_str:<8} "
                  f"{s_str:<8} {ef_str:<7} {c.topsis_score:.3f}  {priority}")

        print("=" * 90)
