"""
run_stage5.py - TE Property Prediction + TOPSIS Multi-Criteria Ranking

ALGORITHM:
    1. Load screening_results.json (CHGNet stability data)
    2. Load safe_candidates.json (gap analysis scores + partner_type)
    3. Merge data: enrich screening results with partner_type from safe_candidates
    4. Run TE property prediction (composition-specific bandgaps + partner-type conductivity)
    5. Run TOPSIS ranking combining all criteria
    6. Save final ranked results

DATA FLOW:
    screening_results.json  ─┐
                              ├─→ merged candidates ─→ TE prediction ─→ TOPSIS ranking
    safe_candidates.json   ─┘
"""

import json
from pathlib import Path

from loguru import logger

from src.screening.te_predictor import TEPredictor, TEProperties
from src.screening.ranker import CandidateRanker


def load_dft_bandgaps(base_dir: Path) -> dict:
    """
    Load DFT-validated bandgaps to override semi-empirical estimates.

    WHY: DFT is the gold standard. If we have DFT results for a MXene,
    those bandgaps should replace any literature estimates. This feeds
    the DFT ground truth back into the TE prediction pipeline.

    RETURNS:
        dict mapping (mxene_formula, termination) → bandgap_ev
        e.g., ("Mo2C", "O") → 0.0  (metallic)
    """
    dft_path = base_dir / "data" / "dft_validation_results.json"
    dft_bandgaps = {}

    if dft_path.exists():
        with open(dft_path) as f:
            dft_results = json.load(f)

        for name, result in dft_results.items():
            # Parse name like "Mo2C_O" → ("Mo2C", "O")
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                formula, termination = parts
                bg = result.get("dft_bandgap")
                if bg is not None:
                    dft_bandgaps[(formula, termination)] = bg
                    print(f"  DFT override: {formula}_{termination} -> bandgap = {bg:.3f} eV"
                          f" ({'metallic' if bg == 0 else 'semiconductor'})")

    return dft_bandgaps


def main():
    base_dir = Path("D:/MXDiscovery")

    # --- Load data ---
    screening_path = base_dir / "data" / "screening_results.json"
    safe_path = base_dir / "data" / "safe_candidates.json"

    with open(screening_path) as f:
        screening_results = json.load(f)
    with open(safe_path) as f:
        safe_candidates = json.load(f)

    logger.info(f"Loaded {len(screening_results)} screening results, {len(safe_candidates)} safe candidates")

    # --- Load DFT bandgaps (if available) to override estimates ---
    print("\n  Loading DFT validation results...")
    dft_bandgaps = load_dft_bandgaps(base_dir)
    if dft_bandgaps:
        # Update the TEPredictor's bandgap table with DFT values
        from src.screening.te_predictor import TEPredictor
        for key, bg in dft_bandgaps.items():
            TEPredictor.BANDGAP_ESTIMATES[key] = bg
        print(f"  Integrated {len(dft_bandgaps)} DFT bandgap(s) into prediction pipeline")
    else:
        print("  No DFT results found — using literature estimates only")

    # --- Build lookup from safe_candidates for partner_type, novelty, synthesizability ---
    safe_lookup = {}
    for sc in safe_candidates:
        key = f"{sc['mxene_formula']}_{sc['termination']}_{sc['composite_partner']}"
        safe_lookup[key] = sc

    # --- Enrich screening results with safe_candidate metadata ---
    for sr in screening_results:
        key = f"{sr['mxene_formula']}_{sr['termination']}_{sr['composite_partner']}"
        sc = safe_lookup.get(key, {})
        sr["partner_type"] = sc.get("partner_type", "bare")
        sr["novelty_score"] = sc.get("novelty_score", 0.5)
        sr["synthesizability"] = sc.get("synthesizability", 0.5)
        sr["overall_gap_score"] = sc.get("overall_score", 0.5)

    # --- Stage 5a: TE Property Prediction ---
    print("\n" + "=" * 70)
    print("  STAGE 5a: THERMOELECTRIC PROPERTY PREDICTION")
    print("=" * 70)

    predictor = TEPredictor(temperature_k=310.0)  # body temperature for wearables

    # Build candidates list for screen_candidates
    candidates_for_te = []
    for sr in screening_results:
        is_stable = sr.get("is_stable", False)
        if isinstance(is_stable, str):
            is_stable = is_stable.lower() == "true"

        candidates_for_te.append({
            "name": sr["name"],
            "mxene_formula": sr["mxene_formula"],
            "termination": sr["termination"],
            "composite_partner": sr["composite_partner"],
            "partner_type": sr["partner_type"],
            "is_stable": is_stable,
            "atoms": None,  # we don't have atoms objects here, use estimation
        })

    te_predictions = predictor.screen_candidates(candidates_for_te, method="estimate")

    print(f"\n  Predicted TE properties for {len(te_predictions)} stable candidates:")
    print(f"  {'Name':<30} {'Eg(eV)':>8} {'S(uV/K)':>8} {'sig(S/cm)':>10} {'PF':>10} {'k(W/mK)':>8} {'ZT':>8}")
    print("  " + "-" * 88)

    for tp in te_predictions:
        bg = f"{tp.bandgap:.3f}" if tp.bandgap else "  -  "
        s = f"{tp.seebeck_coefficient:.1f}" if tp.seebeck_coefficient else "  -  "
        sigma = f"{tp.electrical_conductivity:.1f}" if tp.electrical_conductivity else "  -  "
        pf = f"{tp.power_factor:.1f}" if tp.power_factor else "  -  "
        kappa = f"{tp.thermal_conductivity:.2f}" if tp.thermal_conductivity else "  -  "
        zt = f"{tp.zt_value:.4f}" if tp.zt_value else "  -  "
        print(f"  {tp.name:<30} {bg:>8} {s:>8} {sigma:>8} {pf:>10} {kappa:>8} {zt:>8}")

    # --- Stage 5b: TOPSIS Ranking ---
    print("\n" + "=" * 70)
    print("  STAGE 5b: TOPSIS MULTI-CRITERIA RANKING")
    print("=" * 70)

    ranker = CandidateRanker()

    # Build gap_candidates-like objects for the ranker
    from src.knowledge.gap_analyzer import Candidate
    gap_candidates = []
    for sr in screening_results:
        key = f"{sr['mxene_formula']}_{sr['termination']}_{sr['composite_partner']}"
        sc = safe_lookup.get(key, {})
        gc = Candidate(
            mxene_formula=sr["mxene_formula"],
            m_elements=sc.get("m_elements", [sr["mxene_formula"][:2]]),
            x_element=sc.get("x_element", "C"),
            stoichiometry=sc.get("stoichiometry", "M2X"),
            termination=sr["termination"],
            composite_partner=sr["composite_partner"],
            partner_type=sr.get("partner_type", "bare"),
            novelty_score=sr.get("novelty_score", 0.5),
            synthesizability=sr.get("synthesizability", 0.5),
        )
        gap_candidates.append(gc)

    ranked = ranker.rank(
        candidates=screening_results,
        te_predictions=te_predictions,
        gap_candidates=gap_candidates,
    )

    ranker.print_rankings(ranked)

    # --- Save results ---
    output_path = base_dir / "data" / "final_rankings.json"
    rankings_data = []
    for rc in ranked:
        rankings_data.append({
            "rank": rc.rank,
            "name": rc.name,
            "mxene_formula": rc.mxene_formula,
            "termination": rc.termination,
            "composite_partner": rc.composite_partner,
            "partner_type": rc.partner_type,
            "formation_energy": rc.formation_energy,
            "is_stable": rc.is_stable,
            "seebeck": rc.seebeck,
            "conductivity": rc.conductivity,
            "thermal_conductivity": rc.thermal_cond,
            "power_factor": rc.power_factor,
            "zt": rc.zt,
            "novelty_score": rc.novelty_score,
            "synthesizability": rc.synthesizability,
            "topsis_score": rc.topsis_score,
            "method": rc.method,
            "confidence": rc.confidence,
            "recommendation": rc.recommendation,
        })

    with open(output_path, "w") as f:
        json.dump(rankings_data, f, indent=2)

    print(f"\n  Saved {len(rankings_data)} ranked candidates to {output_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  DISCOVERY SUMMARY")
    print("=" * 70)
    top5 = ranked[:5]
    for rc in top5:
        print(f"\n  #{rc.rank}: {rc.mxene_formula}Tx-{rc.termination} / {rc.composite_partner}")
        print(f"    TOPSIS Score: {rc.topsis_score:.3f}  |  {rc.recommendation}")
        print(f"    Seebeck: {rc.seebeck:.1f} uV/K  |  PF: {rc.power_factor:.1f} uW/cm.K2  |  ZT: {rc.zt:.4f}")
        print(f"    Formation Energy: {rc.formation_energy:.3f} eV/atom  |  Stable: {rc.is_stable}")
        print(f"    Novelty: {rc.novelty_score:.3f}  |  Synthesizability: {rc.synthesizability:.2f}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
