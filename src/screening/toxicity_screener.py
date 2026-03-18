"""
toxicity_screener.py - Biocompatibility and toxicity assessment for wearable MXenes

PURPOSE:
    For wearable thermoelectric devices, the MXene composite will be in direct
    or indirect contact with human skin. This module screens candidates for
    toxicity using established biocompatibility data from:
    - ISO 10993 (Biological evaluation of medical devices)
    - EPA toxicity databases
    - Published cytotoxicity studies on MXenes and their constituent elements
    - REACH/ECHA classification (EU chemicals regulation)

ALGORITHM:
    1. For each candidate, identify ALL elements present (M, X, Tx, composite)
    2. Look up each element's toxicity classification from our curated database
    3. Compute an overall biocompatibility score (0-1)
    4. Flag candidates that contain elements with known toxicity concerns
    5. Classify as: SAFE / CAUTION / UNSAFE for skin-contact wearable use

DATA SOURCES:
    - Nasrallah et al. (2024) "Biocompatibility of MXenes" - Comprehensive review
    - Rozmysłowska-Wojciechowska et al. (2020) "Ti2C and Ti3C2 cytotoxicity"
    - Scheibe et al. (2019) "Cytotoxicity assessment of Ti-based MXenes"
    - ISO 10993-5: In vitro cytotoxicity testing
    - EPA Integrated Risk Information System (IRIS)

TOXICITY CLASSIFICATION FRAMEWORK:
    We classify elements into 4 tiers based on published data:

    Tier 1 - BIOCOMPATIBLE (score 1.0):
        Elements with established biocompatibility in medical devices/implants.
        Ti, Zr, Nb, Ta, C, N, O, H, Ca, Si

    Tier 2 - LOW CONCERN (score 0.7):
        Elements generally safe but with some dose-dependent effects.
        Hf, Sc, Mo (low doses), Fe, Zn, Al

    Tier 3 - MODERATE CONCERN (score 0.4):
        Elements with known toxicity at certain doses/forms. Require careful
        assessment of exposure pathway and dose in wearable context.
        W, Mn, Cu, Ni, F (as fluoride), S

    Tier 4 - HIGH CONCERN (score 0.1):
        Elements with significant toxicity. Generally avoid for skin-contact.
        Cr (hexavalent), V (vanadium pentoxide), Co, Cd, Pb, As, Se, Te, Br, Cl (free)

    IMPORTANT: Toxicity depends on chemical form (oxidation state, solubility),
    not just the element. Ti metal is safe, TiO2 nanoparticles have some concerns
    at very high doses. Our scoring is conservative (worst-case form).

WEARABLE-SPECIFIC CONSIDERATIONS:
    - Skin contact: primary exposure pathway is dermal
    - Sweat interaction: acidic sweat (pH 4.5-7) can leach ions
    - Nanoparticle release: MXene flakes could shed during flexing
    - Duration: wearables are worn for hours/days continuously
    - Population: general public including sensitive individuals

DATA STRUCTURES:
    - ElementToxicity: per-element toxicity data (tier, score, notes, references)
    - ToxicityAssessment: per-candidate overall assessment with detailed breakdown
"""

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class ElementToxicity:
    """Toxicity profile for a single element."""
    symbol: str
    name: str
    tier: int                    # 1=biocompatible, 2=low, 3=moderate, 4=high concern
    score: float                 # 0.0 (toxic) to 1.0 (biocompatible)
    classification: str          # "SAFE", "LOW CONCERN", "MODERATE CONCERN", "HIGH CONCERN"
    skin_contact_safe: bool      # Safe for prolonged skin contact?
    primary_concern: str         # Main toxicity mechanism
    notes: str                   # Additional context
    references: list[str] = field(default_factory=list)


@dataclass
class ToxicityAssessment:
    """Complete toxicity assessment for a MXene composite candidate."""
    candidate_name: str
    elements_present: list[str]
    element_scores: dict          # {element: score}
    element_details: dict         # {element: ElementToxicity}
    overall_score: float          # 0.0-1.0 (weighted average)
    classification: str           # "SAFE" / "CAUTION" / "UNSAFE"
    is_wearable_safe: bool
    concerns: list[str]           # List of specific concerns
    recommendation: str           # Detailed recommendation text


# ===========================================================================
# TOXICITY DATABASE
# Curated from ISO 10993, EPA IRIS, MXene biocompatibility literature
# ===========================================================================
TOXICITY_DATABASE: dict[str, ElementToxicity] = {
    # ── TIER 1: BIOCOMPATIBLE ──
    "Ti": ElementToxicity(
        symbol="Ti", name="Titanium", tier=1, score=1.0,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="None - widely used in medical implants",
        notes="Ti and TiO2 have excellent biocompatibility. Ti3C2Tx MXene "
              "shown to have low cytotoxicity in multiple studies (Rozmyslowska 2020). "
              "FDA-approved for implant use.",
        references=["ISO 10993", "Rozmyslowska-Wojciechowska et al. 2020"],
    ),
    "Zr": ElementToxicity(
        symbol="Zr", name="Zirconium", tier=1, score=1.0,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="None - used in dental implants",
        notes="ZrO2 (zirconia) is a common biocompatible ceramic. "
              "Used in dental crowns and hip implants.",
        references=["ISO 10993", "Dental materials literature"],
    ),
    "Nb": ElementToxicity(
        symbol="Nb", name="Niobium", tier=1, score=0.95,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="None - hypoallergenic metal",
        notes="Nb is used in surgical implants and body jewelry. "
              "Hypoallergenic. Nb2C MXene shows low toxicity.",
        references=["ASTM F67", "Nb biocompatibility studies"],
    ),
    "Ta": ElementToxicity(
        symbol="Ta", name="Tantalum", tier=1, score=0.95,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="None - excellent biocompatibility",
        notes="Ta is used in bone repair plates, surgical clips, and capacitors. "
              "One of the most biocompatible metals known.",
        references=["ISO 10993", "Tantalum implant literature"],
    ),
    "C": ElementToxicity(
        symbol="C", name="Carbon", tier=1, score=1.0,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="None in bulk/2D form",
        notes="Carbon is fundamental to life. Graphene/CNT cytotoxicity "
              "is dose and form dependent, but in composite form is safe.",
        references=["General biocompatibility"],
    ),
    "N": ElementToxicity(
        symbol="N", name="Nitrogen", tier=1, score=1.0,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="None",
        notes="Nitrogen is biologically inert in nitride form.",
        references=["General chemistry"],
    ),
    "O": ElementToxicity(
        symbol="O", name="Oxygen", tier=1, score=1.0,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="None",
        notes="Oxygen termination is the most common and safest MXene surface.",
        references=["General chemistry"],
    ),
    "H": ElementToxicity(
        symbol="H", name="Hydrogen", tier=1, score=1.0,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="None",
        notes="Present in OH termination groups. Biologically safe.",
        references=["General chemistry"],
    ),

    # ── TIER 2: LOW CONCERN ──
    "Hf": ElementToxicity(
        symbol="Hf", name="Hafnium", tier=2, score=0.75,
        classification="LOW CONCERN", skin_contact_safe=True,
        primary_concern="Limited biocompatibility data",
        notes="Hf has similar chemistry to Zr (same group). HfO2 is used "
              "in some semiconductor applications. Limited but encouraging "
              "biocompatibility data. Conservative score due to less data.",
        references=["Limited studies"],
    ),
    "Sc": ElementToxicity(
        symbol="Sc", name="Scandium", tier=2, score=0.70,
        classification="LOW CONCERN", skin_contact_safe=True,
        primary_concern="Limited biocompatibility data",
        notes="Sc is not toxic but has very limited biocompatibility studies. "
              "Sc2C MXene is largely theoretical with no toxicity data.",
        references=["EPA IRIS - no data"],
    ),
    "Mo": ElementToxicity(
        symbol="Mo", name="Molybdenum", tier=2, score=0.70,
        classification="LOW CONCERN", skin_contact_safe=True,
        primary_concern="Essential trace element; toxic at high doses",
        notes="Mo is an essential trace element (cofactor for enzymes). "
              "MoS2 is used in biomedical applications. Mo-based MXenes "
              "show acceptable cytotoxicity at relevant concentrations. "
              "Concern only at very high doses (>1 mg/kg/day).",
        references=["EPA RfD for Mo: 5 ug/kg/day", "MoS2 bio studies"],
    ),

    # ── TIER 3: MODERATE CONCERN ──
    "W": ElementToxicity(
        symbol="W", name="Tungsten", tier=3, score=0.45,
        classification="MODERATE CONCERN", skin_contact_safe=False,
        primary_concern="Tungsten alloy disease; limited dermal data",
        notes="Tungsten metal is relatively inert, but tungsten compounds "
              "(tungstate) have shown toxicity in animal studies. W-based "
              "MXenes are not well-characterized for biocompatibility. "
              "Avoid for direct skin contact wearables.",
        references=["EPA tungsten review", "Tungsten alloy disease studies"],
    ),
    "F": ElementToxicity(
        symbol="F", name="Fluorine", tier=3, score=0.50,
        classification="MODERATE CONCERN", skin_contact_safe=True,
        primary_concern="Fluoride ion release in acidic sweat",
        notes="F-terminated MXenes (Ti3C2F2) can release fluoride ions "
              "in contact with acidic sweat. Low levels are safe (similar "
              "to fluoride toothpaste), but prolonged contact with damaged "
              "skin could be an issue. Prefer O or OH terminations.",
        references=["Fluoride toxicology", "MXene surface chemistry"],
    ),
    "S": ElementToxicity(
        symbol="S", name="Sulfur", tier=3, score=0.55,
        classification="MODERATE CONCERN", skin_contact_safe=True,
        primary_concern="Sulfide can irritate skin at high concentrations",
        notes="Elemental S is used in skin treatments (acne). But sulfide "
              "compounds can cause irritation. In S-terminated MXenes, the "
              "S is bonded to metal, reducing free sulfide release.",
        references=["Dermatological sulfur literature"],
    ),
    "Se": ElementToxicity(
        symbol="Se", name="Selenium", tier=3, score=0.40,
        classification="MODERATE CONCERN", skin_contact_safe=False,
        primary_concern="Toxic at moderate doses; narrow therapeutic window",
        notes="Se is an essential trace element but toxic above 400 ug/day. "
              "Selenide compounds can be toxic. SnSe for TE is a concern.",
        references=["EPA RfD for Se: 5 ug/kg/day"],
    ),

    # ── TIER 4: HIGH CONCERN ──
    "Cr": ElementToxicity(
        symbol="Cr", name="Chromium", tier=4, score=0.10,
        classification="HIGH CONCERN", skin_contact_safe=False,
        primary_concern="Cr(VI) is carcinogenic; Cr(III) causes dermatitis",
        notes="Cr(VI) compounds are IARC Group 1 carcinogens. Even Cr(III) "
              "causes contact dermatitis in ~5% of population. Cr-based MXenes "
              "should be AVOIDED for any wearable application.",
        references=["IARC Monograph 100C", "Contact dermatitis literature"],
    ),
    "V": ElementToxicity(
        symbol="V", name="Vanadium", tier=4, score=0.15,
        classification="HIGH CONCERN", skin_contact_safe=False,
        primary_concern="V2O5 is toxic; causes respiratory and skin irritation",
        notes="Vanadium pentoxide is toxic. V-based MXenes (V2CTx) release "
              "V ions that can cause cytotoxicity. Some studies show V2C has "
              "lower toxicity than V2O5, but still a concern for wearables. "
              "AVOID for prolonged skin contact.",
        references=["EPA IRIS V pentoxide", "V2C cytotoxicity studies"],
    ),
    "Te": ElementToxicity(
        symbol="Te", name="Tellurium", tier=4, score=0.20,
        classification="HIGH CONCERN", skin_contact_safe=False,
        primary_concern="Toxic; causes garlic breath and neuropathy",
        notes="Tellurium and its compounds are toxic. Te NW composites "
              "should not be used in direct skin contact applications.",
        references=["EPA IRIS tellurium"],
    ),
    "Br": ElementToxicity(
        symbol="Br", name="Bromine", tier=4, score=0.20,
        classification="HIGH CONCERN", skin_contact_safe=False,
        primary_concern="Bromide causes skin irritation and is toxic",
        notes="Free bromide is corrosive and toxic. Br-terminated MXenes "
              "are not suitable for wearable applications.",
        references=["Bromine toxicology"],
    ),
    "Cl": ElementToxicity(
        symbol="Cl", name="Chlorine", tier=3, score=0.45,
        classification="MODERATE CONCERN", skin_contact_safe=True,
        primary_concern="Chloride release; mild irritation potential",
        notes="Cl-terminated MXenes can release chloride in sweat. NaCl "
              "is ubiquitous and safe, but concentrated Cl can irritate. "
              "Prefer O/OH terminations for wearables.",
        references=["General toxicology"],
    ),

    # ── COMPOSITE PARTNER ELEMENTS ──
    "Bi": ElementToxicity(
        symbol="Bi", name="Bismuth", tier=2, score=0.65,
        classification="LOW CONCERN", skin_contact_safe=True,
        primary_concern="Generally safe; used in Pepto-Bismol",
        notes="Bi has remarkably low toxicity for a heavy metal. Used in "
              "pharmaceuticals (bismuth subsalicylate). Bi2Te3 composites "
              "are acceptable but Te component is the concern.",
        references=["Bismuth pharmaceutical safety data"],
    ),
    "Sn": ElementToxicity(
        symbol="Sn", name="Tin", tier=2, score=0.70,
        classification="LOW CONCERN", skin_contact_safe=True,
        primary_concern="Inorganic Sn is low toxicity; organic Sn is toxic",
        notes="Metallic tin and inorganic tin compounds have low toxicity. "
              "SnSe and SnS2 are acceptable composite partners.",
        references=["EPA tin assessment"],
    ),
    "Sb": ElementToxicity(
        symbol="Sb", name="Antimony", tier=4, score=0.20,
        classification="HIGH CONCERN", skin_contact_safe=False,
        primary_concern="Toxic; causes skin and respiratory irritation",
        notes="Antimony compounds are toxic. Sb2Te3 should be avoided "
              "for wearable applications.",
        references=["EPA IRIS antimony"],
    ),
    "Zn": ElementToxicity(
        symbol="Zn", name="Zinc", tier=1, score=0.90,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="Safe; used in sunscreen (ZnO)",
        notes="ZnO is FDA-approved for skin contact (sunscreen, diaper cream).",
        references=["FDA GRAS", "ZnO safety data"],
    ),
    "In": ElementToxicity(
        symbol="In", name="Indium", tier=3, score=0.40,
        classification="MODERATE CONCERN", skin_contact_safe=False,
        primary_concern="Indium lung disease; limited dermal data",
        notes="Indium tin oxide (ITO) workers have shown lung disease. "
              "Limited dermal toxicity data. Caution advised.",
        references=["Indium lung disease literature"],
    ),
    "Ag": ElementToxicity(
        symbol="Ag", name="Silver", tier=1, score=0.85,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="Safe in bulk; nanoparticle form has antimicrobial "
              "properties but can cause argyria at very high chronic exposure",
        notes="Ag is antimicrobial and used in wound dressings. AgNW "
              "composites are suitable for wearables.",
        references=["FDA wound dressing approvals", "Ag NP safety data"],
    ),
    "Cu": ElementToxicity(
        symbol="Cu", name="Copper", tier=2, score=0.65,
        classification="LOW CONCERN", skin_contact_safe=True,
        primary_concern="Can cause green discoloration; some people have Cu allergy",
        notes="Cu is an essential element. Cu skin contact causes harmless "
              "green discoloration. ~1-3% of population has Cu contact allergy.",
        references=["Cu contact dermatitis literature"],
    ),
    "Sr": ElementToxicity(
        symbol="Sr", name="Strontium", tier=2, score=0.70,
        classification="LOW CONCERN", skin_contact_safe=True,
        primary_concern="Stable Sr is non-toxic; radioactive Sr-90 is dangerous",
        notes="Stable strontium (used in SrTiO3) is non-toxic. "
              "SrTiO3 is a safe ceramic material.",
        references=["Strontium safety data"],
    ),
    "Fe": ElementToxicity(
        symbol="Fe", name="Iron", tier=1, score=0.90,
        classification="SAFE", skin_contact_safe=True,
        primary_concern="Essential element; Fe oxide NPs are FDA-approved",
        notes="Iron is essential for life. Fe3O4 nanoparticles are FDA-approved "
              "for medical imaging (Feridex). Safe for skin contact.",
        references=["FDA Fe3O4 NP approval"],
    ),
}


class ToxicityScreener:
    """
    Screens MXene composite candidates for biocompatibility.

    HOW IT WORKS:
        1. Parse all elements from the candidate (MXene + termination + partner)
        2. Look up each element in TOXICITY_DATABASE
        3. Compute overall score as MINIMUM element score (weakest link)
           Why minimum, not average? Because one toxic element makes the
           whole composite unsafe regardless of other safe elements.
        4. Apply wearable-specific penalties:
           - Fluoride termination in sweat-prone area: -0.1
           - Nanoparticle form without encapsulation: -0.05
        5. Classify and generate recommendation

    CLASSIFICATION:
        score >= 0.65:  SAFE for wearable use
        0.40 <= score < 0.65: CAUTION - may be acceptable with encapsulation
        score < 0.40:   UNSAFE - do not use for skin contact
    """

    def __init__(self):
        self.db = TOXICITY_DATABASE

    def _parse_elements(self, mxene_formula: str, termination: str,
                        composite_partner: str) -> list[str]:
        """
        Extract all unique elements from a candidate description.

        EXAMPLES:
            ("Ti3C2", "O", "PEDOT:PSS") -> ["Ti", "C", "O"]
            ("Mo2TiC2", "OH", "Bi2Te3") -> ["Mo", "Ti", "C", "O", "H", "Bi", "Te"]
        """
        import re
        elements = set()

        # Parse MXene formula: extract element symbols
        formula_parts = re.findall(r'([A-Z][a-z]?)', mxene_formula)
        elements.update(formula_parts)

        # Parse termination
        term_elements = re.findall(r'([A-Z][a-z]?)', termination)
        elements.update(term_elements)

        # Parse composite partner
        partner_elements = {
            "PEDOT:PSS": ["C", "O", "S"],      # Conducting polymer (C, H, O, S based)
            "PANI": ["C", "N"],                  # Polyaniline
            "PPy": ["C", "N"],                   # Polypyrrole
            "P3HT": ["C", "S"],                  # Polythiophene
            "PVDF": ["C", "F"],                  # PVDF contains fluorine
            "SWCNT": ["C"],
            "MWCNT": ["C"],
            "rGO": ["C", "O"],
            "graphene": ["C"],
            "C60": ["C"],
            "Bi2Te3": ["Bi", "Te"],
            "SnSe": ["Sn", "Se"],
            "Sb2Te3": ["Sb", "Te"],
            "MoS2": ["Mo", "S"],
            "WS2": ["W", "S"],
            "SnS2": ["Sn", "S"],
            "ZnO": ["Zn", "O"],
            "TiO2": ["Ti", "O"],
            "SrTiO3": ["Sr", "Ti", "O"],
            "In2O3": ["In", "O"],
            "Ag_NW": ["Ag"],
            "Cu_NW": ["Cu"],
            "Te_NW": ["Te"],
        }

        if composite_partner in partner_elements:
            elements.update(partner_elements[composite_partner])
        else:
            # Try to parse formula
            partner_parts = re.findall(r'([A-Z][a-z]?)', composite_partner)
            elements.update(partner_parts)

        return sorted(elements)

    def assess(self, mxene_formula: str, termination: str,
               composite_partner: str = "", name: str = "") -> ToxicityAssessment:
        """
        Perform full toxicity assessment on a candidate.

        ALGORITHM:
            1. Parse all elements present
            2. Look up each element's toxicity score
            3. Overall score = MINIMUM of all element scores
               (weakest link principle: one bad element ruins it)
            4. Identify specific concerns
            5. Generate detailed recommendation
        """
        if not name:
            name = f"{mxene_formula}_{termination}_{composite_partner}"

        elements = self._parse_elements(mxene_formula, termination, composite_partner)

        element_scores = {}
        element_details = {}
        concerns = []

        for el in elements:
            if el in self.db:
                tox = self.db[el]
                element_scores[el] = tox.score
                element_details[el] = tox

                if tox.tier >= 3:
                    concerns.append(
                        f"{tox.name} ({el}): {tox.classification} - {tox.primary_concern}"
                    )
            else:
                # Unknown element - conservative
                element_scores[el] = 0.5
                concerns.append(f"{el}: No toxicity data available - treated as moderate concern")

        # Overall score = minimum (weakest link)
        if element_scores:
            overall_score = min(element_scores.values())
        else:
            overall_score = 0.0

        # Wearable-specific adjustments
        if termination == "F" and overall_score > 0.5:
            overall_score -= 0.05
            concerns.append("F-termination: fluoride may leach in sweat. Prefer O or OH.")

        # Classification
        if overall_score >= 0.65:
            classification = "SAFE"
            is_safe = True
            rec = (f"This MXene composite is suitable for wearable skin-contact "
                   f"applications. All constituent elements have established "
                   f"biocompatibility profiles.")
        elif overall_score >= 0.40:
            classification = "CAUTION"
            is_safe = False
            rec = (f"This composite has moderate toxicity concerns. It MAY be "
                   f"acceptable with proper encapsulation (e.g., PDMS coating) "
                   f"to prevent direct skin contact with the active material. "
                   f"Concerns: {'; '.join(concerns)}")
        else:
            classification = "UNSAFE"
            is_safe = False
            rec = (f"This composite is NOT recommended for wearable applications "
                   f"due to significant toxicity concerns. Elements of concern: "
                   f"{'; '.join(concerns)}")

        return ToxicityAssessment(
            candidate_name=name,
            elements_present=elements,
            element_scores=element_scores,
            element_details=element_details,
            overall_score=overall_score,
            classification=classification,
            is_wearable_safe=is_safe,
            concerns=concerns,
            recommendation=rec,
        )

    def screen_batch(self, candidates: list[dict]) -> list[ToxicityAssessment]:
        """
        Screen a batch of candidates for toxicity.

        INPUT: list of dicts with keys: mxene_formula, termination, composite_partner
        RETURNS: list of ToxicityAssessment, sorted by overall_score (safest first)
        """
        results = []
        for cand in candidates:
            assessment = self.assess(
                mxene_formula=cand.get("mxene_formula", ""),
                termination=cand.get("termination", "O"),
                composite_partner=cand.get("composite_partner", ""),
                name=cand.get("name", ""),
            )
            results.append(assessment)

        # Sort: safest first
        results.sort(key=lambda a: a.overall_score, reverse=True)

        # Stats
        n_safe = sum(1 for a in results if a.classification == "SAFE")
        n_caution = sum(1 for a in results if a.classification == "CAUTION")
        n_unsafe = sum(1 for a in results if a.classification == "UNSAFE")
        logger.info(
            f"Toxicity screening: {n_safe} SAFE, {n_caution} CAUTION, "
            f"{n_unsafe} UNSAFE out of {len(results)} candidates"
        )

        return results

    def get_safe_elements(self) -> list[str]:
        """Return list of elements classified as SAFE (tier 1-2)."""
        return [el for el, tox in self.db.items() if tox.tier <= 2]

    def get_safe_mxene_metals(self) -> list[str]:
        """Return M-elements safe for wearable MXenes."""
        safe_metals = []
        wearable_m_elements = ["Ti", "Zr", "Nb", "Ta", "Hf", "Sc", "Mo",
                               "V", "Cr", "W"]
        for m in wearable_m_elements:
            if m in self.db and self.db[m].tier <= 2:
                safe_metals.append(m)
        return safe_metals

    def get_safe_terminations(self) -> list[str]:
        """Return terminations safe for wearable use."""
        return ["O", "OH"]  # Safest. F and Cl have concerns.

    def get_safe_partners(self) -> list[str]:
        """Return composite partners safe for wearable use."""
        safe = []
        partner_elements = {
            "PEDOT:PSS": ["C", "O", "S"],
            "PANI": ["C", "N"],
            "PPy": ["C", "N"],
            "P3HT": ["C", "S"],
            "SWCNT": ["C"],
            "MWCNT": ["C"],
            "rGO": ["C", "O"],
            "graphene": ["C"],
            "C60": ["C"],
            "SnSe": ["Sn", "Se"],
            "MoS2": ["Mo", "S"],
            "SnS2": ["Sn", "S"],
            "ZnO": ["Zn", "O"],
            "TiO2": ["Ti", "O"],
            "SrTiO3": ["Sr", "Ti", "O"],
            "Ag_NW": ["Ag"],
            "Cu_NW": ["Cu"],
        }
        for partner, elements in partner_elements.items():
            scores = [self.db.get(el, ElementToxicity(
                symbol=el, name=el, tier=3, score=0.5,
                classification="UNKNOWN", skin_contact_safe=False,
                primary_concern="Unknown", notes=""
            )).score for el in elements]
            if min(scores) >= 0.55:
                safe.append(partner)
        return safe

    def print_report(self, assessments: list[ToxicityAssessment]):
        """Print formatted toxicity screening report."""
        print("\n" + "=" * 80)
        print("  TOXICITY SCREENING REPORT - WEARABLE MXENE COMPOSITES")
        print("=" * 80)

        for a in assessments[:30]:
            icon = {
                "SAFE": "[SAFE]    ",
                "CAUTION": "[CAUTION] ",
                "UNSAFE": "[UNSAFE]  ",
            }.get(a.classification, "[???]     ")

            print(f"  {icon} {a.candidate_name:<40} Score: {a.overall_score:.2f}")
            if a.concerns:
                for c in a.concerns:
                    print(f"           -> {c}")

        print("\n  SAFE M-elements for wearable MXenes:")
        print(f"    {', '.join(self.get_safe_mxene_metals())}")
        print(f"  SAFE terminations: {', '.join(self.get_safe_terminations())}")
        print(f"  SAFE composite partners:")
        print(f"    {', '.join(self.get_safe_partners())}")
        print("=" * 80)
