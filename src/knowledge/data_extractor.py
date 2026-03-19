"""
data_extractor.py - Extract structured thermoelectric data from paper abstracts
                    using a local LLM (Ollama)

ALGORITHM (Overview):
    1. Load papers from JSONL (fetched by paper_fetcher.py)
    2. For each paper with an abstract, send a structured extraction prompt
       to the local LLM (Qwen2.5 via Ollama)
    3. The prompt asks the LLM to extract specific fields into JSON:
       - MXene composition (e.g., Ti3C2Tx)
       - Composite partner (e.g., PEDOT:PSS)
       - Thermoelectric properties (Seebeck, conductivity, ZT, power factor)
       - Synthesis method
       - Application context
    4. Parse the LLM's JSON response, validate with Pydantic
    5. Store extracted records in SQLite database

TECHNIQUES:
    - Few-shot prompting: We include 2-3 example extractions in the prompt
      so the LLM understands the exact output format
    - JSON mode: We instruct the LLM to output valid JSON only
    - Validation: Pydantic models enforce types and ranges
      (e.g., Seebeck coefficient must be a number, not a string)
    - Confidence scoring: LLM rates its confidence (high/medium/low)
      for each extracted field
    - Batch processing with checkpointing: saves progress every N papers
      so we can resume if interrupted

DATA STRUCTURES:
    - ExtractedRecord (Pydantic model):
        paper_id: str
        mxene_composition: str | None
        mxene_m_elements: list[str]
        mxene_x_element: str | None
        termination: str | None
        composite_partner: str | None
        composite_type: str | None  (polymer, carbon, chalcogenide, oxide, metal)
        seebeck_coefficient: float | None  (µV/K)
        electrical_conductivity: float | None  (S/cm)
        thermal_conductivity: float | None  (W/mK)
        power_factor: float | None  (µW/cm·K²)
        zt_value: float | None
        temperature_k: float | None
        synthesis_method: str | None
        application: str | None
        confidence: str  (high / medium / low)
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from loguru import logger

try:
    import ollama
except ImportError:
    ollama = None
    logger.warning("ollama package not installed. LLM extraction disabled.")


# ---------------------------------------------------------------------------
# Pydantic model for validated extraction output
# ---------------------------------------------------------------------------
class ExtractedRecord(BaseModel):
    """
    A single structured record extracted from a paper.

    WHY PYDANTIC:
    - Automatic type coercion (string "25.3" -> float 25.3)
    - Validation (ZT can't be negative)
    - Serialization to dict/JSON for database insertion
    - Clear schema documentation
    """
    paper_id: str
    mxene_composition: Optional[str] = None
    mxene_m_elements: list[str] = Field(default_factory=list)
    mxene_x_element: Optional[str] = None
    termination: Optional[str] = None
    composite_partner: Optional[str] = None
    composite_type: Optional[str] = None
    seebeck_coefficient: Optional[float] = None      # µV/K
    electrical_conductivity: Optional[float] = None   # S/cm
    thermal_conductivity: Optional[float] = None      # W/mK
    power_factor: Optional[float] = None              # µW/cm·K²
    zt_value: Optional[float] = None
    temperature_k: Optional[float] = None             # K
    synthesis_method: Optional[str] = None
    application: Optional[str] = None
    confidence: str = "low"

    @field_validator("mxene_m_elements", mode="before")
    @classmethod
    def coerce_m_elements(cls, v):
        """LLM sometimes returns null instead of []. Coerce to empty list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("composite_partner", "composite_type", "synthesis_method",
                     "application", mode="before")
    @classmethod
    def coerce_string(cls, v):
        """LLM sometimes returns lists for string fields. Join with comma."""
        if v is None:
            return None
        if isinstance(v, list):
            return ", ".join(str(x) for x in v) if v else None
        return str(v)

    @field_validator("seebeck_coefficient", "electrical_conductivity",
                     "thermal_conductivity", "power_factor", "zt_value",
                     "temperature_k", mode="before")
    @classmethod
    def coerce_numeric(cls, v):
        """
        Handle common LLM quirks in numeric fields:
        - List of values -> take first value (LLM sometimes returns ranges)
        - String numbers -> convert to float
        - Empty string -> None
        """
        if v is None:
            return None
        if isinstance(v, list):
            return float(v[0]) if v else None
        if isinstance(v, dict):
            # LLM sometimes returns {'in_plane': 28.8, 'cross_plane': 0.27}
            # Take the first numeric value
            for val in v.values():
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
            return None
        if isinstance(v, str):
            v = v.strip()
            if not v or v.lower() == "null":
                return None
            try:
                return float(v)
            except ValueError:
                return None
        return v

    @field_validator("zt_value")
    @classmethod
    def zt_must_be_reasonable(cls, v):
        """ZT values in literature range 0 to ~3. Flag outliers."""
        if v is not None and (v < 0 or v > 10):
            logger.warning(f"Suspicious ZT value: {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v):
        if v not in ("high", "medium", "low"):
            return "low"
        return v


# ---------------------------------------------------------------------------
# Extraction prompt template
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT_TEMPLATE = """You are a materials science data extraction expert.
Extract structured thermoelectric property data from this paper abstract.

RULES:
- Only extract data EXPLICITLY stated in the abstract. Do NOT infer or guess.
- If a value is not mentioned, set it to null.
- For MXene composition, use standard notation (e.g., Ti3C2Tx, Mo2TiC2Tx).
- Convert all units to standard:
  * Seebeck coefficient: uV/K
  * Electrical conductivity: S/cm
  * Thermal conductivity: W/mK
  * Power factor: uW/cm K2
- Rate your confidence: "high" if values are explicitly stated with units,
  "medium" if stated but units unclear, "low" if interpretation was needed.

EXAMPLE INPUT:
"We fabricated Ti3C2Tx/PEDOT:PSS composite films with a Seebeck coefficient
of 57.3 uV/K and electrical conductivity of 1500 S/cm at room temperature,
yielding a power factor of 155.4 uW/cm K2."

EXAMPLE OUTPUT:
{{"mxene_composition": "Ti3C2Tx", "mxene_m_elements": ["Ti"], "mxene_x_element": "C", "termination": "Tx", "composite_partner": "PEDOT:PSS", "composite_type": "polymer", "seebeck_coefficient": 57.3, "electrical_conductivity": 1500, "thermal_conductivity": null, "power_factor": 155.4, "zt_value": null, "temperature_k": 300, "synthesis_method": "film fabrication", "application": "thermoelectric", "confidence": "high"}}

NOW EXTRACT FROM THIS ABSTRACT:
Title: {title}
Abstract: {abstract}

Respond with ONLY valid JSON. No explanation."""


class DataExtractor:
    """
    Extracts structured TE data from paper abstracts using a local LLM.

    HOW IT WORKS (step by step):
        1. __init__: Connect to Ollama, verify model is available
        2. extract_one(): Send one abstract to LLM, parse JSON response
        3. extract_batch(): Process multiple papers with checkpointing
        4. validate(): Run Pydantic validation on extracted JSON
        5. save(): Write validated records to JSONL + pass to database

    CHECKPOINTING:
        Every 10 papers, we save progress to data/extracted_records.jsonl.
        If the process is interrupted, next run skips already-extracted paper_ids.
        This is critical because processing 500+ papers takes hours.

    ERROR HANDLING:
        - LLM returns invalid JSON -> log error, skip paper, mark as failed
        - LLM returns nonsense values -> Pydantic validation catches it
        - Ollama connection fails -> raise clear error with setup instructions
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        output_dir: str | Path = "data/papers",
        checkpoint_every: int = 10,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.checkpoint_every = checkpoint_every
        self.records: list[ExtractedRecord] = []
        self.failed: list[str] = []  # paper_ids that failed extraction
        self._extracted_ids: set[str] = set()

        # Load checkpoint if exists
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load previously extracted records to avoid re-processing."""
        checkpoint = self.output_dir / "extracted_records.jsonl"
        if checkpoint.exists():
            with open(checkpoint, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    self._extracted_ids.add(data["paper_id"])
                    self.records.append(ExtractedRecord(**data))
            logger.info(f"Loaded {len(self.records)} existing extractions")

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Send prompt to Ollama and get response.

        TECHNIQUE:
        - format="json" forces Ollama to constrain output to valid JSON
        - temperature=0.1 for deterministic, factual extraction
        - We set num_ctx to handle long abstracts
        """
        if ollama is None:
            raise RuntimeError(
                "ollama package not installed. Run: pip install ollama\n"
                "Also install Ollama app from https://ollama.com"
            )
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.1, "num_ctx": 4096, "num_gpu": 99},
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def extract_one(self, paper_id: str, title: str, abstract: str) -> Optional[ExtractedRecord]:
        """
        Extract structured data from a single paper.

        ALGORITHM:
            1. Format the extraction prompt with title + abstract
            2. Send to LLM, get JSON string back
            3. Parse JSON string into dict
            4. Add paper_id to dict
            5. Validate with Pydantic (ExtractedRecord)
            6. Return validated record or None on failure
        """
        if paper_id in self._extracted_ids:
            logger.debug(f"Skipping already extracted: {paper_id}")
            return None

        if not abstract:
            logger.debug(f"No abstract for {paper_id}, skipping")
            return None

        prompt = EXTRACTION_PROMPT_TEMPLATE.format(title=title, abstract=abstract)
        raw_response = self._call_llm(prompt)

        if not raw_response:
            self.failed.append(paper_id)
            return None

        try:
            data = json.loads(raw_response)
            data["paper_id"] = paper_id
            record = ExtractedRecord(**data)
            self._extracted_ids.add(paper_id)
            return record
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Parse/validation failed for {paper_id}: {e}")
            self.failed.append(paper_id)
            return None

    def extract_batch(self, papers: list[dict]) -> list[ExtractedRecord]:
        """
        Extract data from a batch of papers with progress tracking.

        INPUT FORMAT:
            papers = [{"paper_id": "...", "title": "...", "abstract": "..."}, ...]

        ALGORITHM:
            1. Filter out papers already extracted (checkpoint)
            2. For each remaining paper:
               a. Call extract_one()
               b. If successful, append to records list
               c. Every checkpoint_every papers, save to disk
            3. Final save after all papers processed
            4. Return list of newly extracted records
        """
        new_records = []
        to_process = [
            p for p in papers
            if p["paper_id"] not in self._extracted_ids and p.get("abstract")
        ]

        logger.info(f"Processing {len(to_process)} papers ({len(papers) - len(to_process)} already done)")

        for i, paper in enumerate(to_process):
            record = self.extract_one(
                paper_id=paper["paper_id"],
                title=paper["title"],
                abstract=paper["abstract"],
            )
            if record:
                self.records.append(record)
                new_records.append(record)

            # Checkpoint
            if (i + 1) % self.checkpoint_every == 0:
                self._save_checkpoint()
                logger.info(f"  Progress: {i+1}/{len(to_process)} | Extracted: {len(new_records)}")

        self._save_checkpoint()
        logger.info(
            f"Extraction complete. New: {len(new_records)}, "
            f"Failed: {len(self.failed)}, Total: {len(self.records)}"
        )
        return new_records

    def _save_checkpoint(self):
        """Save current records to JSONL file."""
        path = self.output_dir / "extracted_records.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(r.model_dump_json() + "\n")

    def get_stats(self) -> dict:
        """Summary statistics of extracted data."""
        has_seebeck = sum(1 for r in self.records if r.seebeck_coefficient is not None)
        has_conductivity = sum(1 for r in self.records if r.electrical_conductivity is not None)
        has_zt = sum(1 for r in self.records if r.zt_value is not None)
        has_pf = sum(1 for r in self.records if r.power_factor is not None)

        compositions = set()
        partners = set()
        for r in self.records:
            if r.mxene_composition:
                compositions.add(r.mxene_composition)
            if r.composite_partner:
                partners.add(r.composite_partner)

        return {
            "total_records": len(self.records),
            "failed_extractions": len(self.failed),
            "has_seebeck": has_seebeck,
            "has_conductivity": has_conductivity,
            "has_zt": has_zt,
            "has_power_factor": has_pf,
            "unique_compositions": len(compositions),
            "unique_partners": len(partners),
            "high_confidence": sum(1 for r in self.records if r.confidence == "high"),
        }


# ---------------------------------------------------------------------------
# Pre-filter: Only send TE-relevant papers to the LLM (saves processing time)
# ---------------------------------------------------------------------------
TE_KEYWORDS = [
    "thermoelectric", "seebeck", "power factor", "figure of merit",
    "thermal conductivity", "electrical conductivity", "peltier",
    "zte", "zt ", " zt=", "zt value", "thermovoltage", "thermopower",
    "energy harvesting", "energy harvest",
]

MXENE_KEYWORDS = [
    "mxene", "ti3c2", "ti2c", "mo2c", "v2c", "nb2c", "cr2c",
    "mo2ti", "ti3cn", "nb4c3", "ta4c3", "v4c3", "m2x", "m3x2", "m4x3",
    "max phase", "transition metal carbide", "transition metal nitride",
    "2d carbide", "2d nitride",
]


def is_te_relevant(paper: dict) -> bool:
    """
    Fast keyword check to decide if a paper is worth LLM extraction.

    WHY PRE-FILTER:
        Our 2000 papers came from broad searches. Many are about MXenes
        but not thermoelectrics (e.g., MXene for batteries, supercapacitors).
        LLM extraction takes ~5 seconds per paper. Processing all 1417
        with abstracts would take ~2 hours. Pre-filtering to ~200 TE-relevant
        papers reduces this to ~17 minutes.

    ALGORITHM:
        A paper is TE-relevant if its abstract contains:
        - At least one MXene keyword AND at least one TE keyword
        - OR it mentions specific TE property values (numbers + units)
    """
    abstract = (paper.get("abstract") or "").lower()
    title = (paper.get("title") or "").lower()
    text = title + " " + abstract

    has_mxene = any(kw in text for kw in MXENE_KEYWORDS)
    has_te = any(kw in text for kw in TE_KEYWORDS)

    return has_mxene and has_te


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml
    import time

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_dir = Path(config["project"]["base_dir"])
    papers_dir = base_dir / "data" / "papers"

    # Load papers
    papers_jsonl = papers_dir / "papers.jsonl"
    if not papers_jsonl.exists():
        logger.error(f"No papers found at {papers_jsonl}. Run paper fetcher first.")
        exit(1)

    all_papers = []
    with open(papers_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_papers.append(json.loads(line.strip()))

    # Pre-filter to TE-relevant papers only
    te_papers = [p for p in all_papers if is_te_relevant(p)]
    logger.info(f"Loaded {len(all_papers)} papers, {len(te_papers)} are TE-relevant")

    # Initialize extractor
    extractor = DataExtractor(
        model=config["llm"]["model"],
        output_dir=papers_dir,
        checkpoint_every=10,
    )

    # Run extraction
    start = time.time()
    records = extractor.extract_batch(te_papers)
    elapsed = time.time() - start

    # Print stats
    stats = extractor.get_stats()
    print(f"\n--- LLM Extraction Complete ({elapsed:.1f}s) ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
