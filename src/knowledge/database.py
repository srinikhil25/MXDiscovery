"""
database.py - SQLite database for structured MXene thermoelectric knowledge

ALGORITHM:
    This module provides a relational database layer over the extracted data.
    Instead of querying flat JSON files, we store structured records in SQLite
    which enables:
    - Fast SQL queries (find all Ti3C2Tx composites with Seebeck > 50 µV/K)
    - JOIN operations (link papers to their extracted data)
    - Aggregation (average ZT per MXene type)
    - Gap analysis queries (which compositions have NO reported data?)

DATA STRUCTURES (SQL Tables):
    1. papers:        Raw paper metadata (title, authors, year, citations)
    2. te_records:    Extracted thermoelectric property records
    3. compositions:  Unique MXene compositions encountered
    4. partners:      Unique composite partners encountered
    5. screening:     ML screening results (stability, predicted properties)

TECHNIQUES:
    - SQLAlchemy ORM for type-safe database operations
    - WAL (Write-Ahead Logging) mode for concurrent read/write
    - Indexes on frequently queried columns (composition, partner, year)
    - Upsert pattern: INSERT OR REPLACE to handle re-extraction gracefully

WHY SQLITE:
    - Zero configuration, no server process
    - Single file, easy to backup and share
    - Handles millions of rows efficiently
    - Python has built-in support
    - Sufficient for our scale (thousands, not billions of records)
"""

import sqlite3
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from loguru import logger


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
-- Papers table: stores metadata for each paper
CREATE TABLE IF NOT EXISTS papers (
    paper_id    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    abstract    TEXT,
    year        INTEGER,
    authors     TEXT,           -- JSON array as string
    doi         TEXT,
    venue       TEXT,
    citations   INTEGER DEFAULT 0,
    has_pdf     INTEGER DEFAULT 0,
    tldr        TEXT
);

-- Thermoelectric records: one row per extracted measurement
-- A single paper can have multiple records (different samples/conditions)
CREATE TABLE IF NOT EXISTS te_records (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id                TEXT NOT NULL,
    mxene_composition       TEXT,
    mxene_m_elements        TEXT,       -- JSON array: ["Ti"] or ["Mo","Ti"]
    mxene_x_element         TEXT,       -- "C" or "N"
    termination             TEXT,       -- "O", "OH", "F", "Tx", etc.
    composite_partner       TEXT,
    composite_type          TEXT,       -- polymer/carbon/chalcogenide/oxide/metal
    seebeck_coefficient     REAL,       -- µV/K
    electrical_conductivity REAL,       -- S/cm
    thermal_conductivity    REAL,       -- W/mK
    power_factor            REAL,       -- µW/cm·K²
    zt_value                REAL,
    temperature_k           REAL,       -- K
    synthesis_method        TEXT,
    application             TEXT,
    confidence              TEXT DEFAULT 'low',
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

-- Screening results from ML potentials (CHGNet/MACE)
CREATE TABLE IF NOT EXISTS screening_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    mxene_composition       TEXT NOT NULL,
    composite_partner       TEXT,
    structure_file          TEXT,       -- path to CIF/POSCAR
    formation_energy        REAL,       -- eV/atom
    energy_above_hull       REAL,       -- eV/atom
    is_stable               INTEGER,    -- 1=stable, 0=unstable
    predicted_seebeck       REAL,
    predicted_conductivity  REAL,
    predicted_kappa         REAL,
    predicted_zt            REAL,
    ml_model                TEXT,       -- "chgnet" or "mace"
    relaxed_structure_file  TEXT,
    notes                   TEXT,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_te_composition ON te_records(mxene_composition);
CREATE INDEX IF NOT EXISTS idx_te_partner ON te_records(composite_partner);
CREATE INDEX IF NOT EXISTS idx_te_year ON te_records(paper_id);
CREATE INDEX IF NOT EXISTS idx_screening_comp ON screening_results(mxene_composition);
CREATE INDEX IF NOT EXISTS idx_screening_partner ON screening_results(composite_partner);
"""


class MXeneDatabase:
    """
    SQLite database manager for MXene thermoelectric data.

    HOW IT WORKS:
        1. __init__: Opens/creates SQLite DB, runs schema migration
        2. insert_paper(): Add paper metadata
        3. insert_te_record(): Add extracted TE property record
        4. insert_screening(): Add ML screening result
        5. query_*(): Various query methods for analysis
        6. gap_analysis(): Find unexplored composition-partner combinations

    USAGE:
        db = MXeneDatabase("data/database/mxene_knowledge.db")
        db.insert_paper(paper_id="abc", title="...", ...)
        results = db.query_by_composition("Ti3C2Tx")
    """

    def __init__(self, db_path: str | Path = "data/database/mxene_knowledge.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # dict-like row access
        self.conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def insert_paper(self, paper_id: str, title: str, abstract: str = None,
                     year: int = None, authors: str = None, doi: str = None,
                     venue: str = None, citations: int = 0, has_pdf: bool = False,
                     tldr: str = None):
        """Insert or update a paper record."""
        self.conn.execute(
            """INSERT OR REPLACE INTO papers
               (paper_id, title, abstract, year, authors, doi, venue, citations, has_pdf, tldr)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (paper_id, title, abstract, year, authors, doi, venue, citations,
             1 if has_pdf else 0, tldr),
        )
        self.conn.commit()

    def insert_te_record(self, record: dict):
        """
        Insert an extracted thermoelectric record.

        INPUT: dict matching ExtractedRecord fields from data_extractor.py
        """
        self.conn.execute(
            """INSERT INTO te_records
               (paper_id, mxene_composition, mxene_m_elements, mxene_x_element,
                termination, composite_partner, composite_type,
                seebeck_coefficient, electrical_conductivity, thermal_conductivity,
                power_factor, zt_value, temperature_k,
                synthesis_method, application, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.get("paper_id"),
                record.get("mxene_composition"),
                str(record.get("mxene_m_elements", [])),
                record.get("mxene_x_element"),
                record.get("termination"),
                record.get("composite_partner"),
                record.get("composite_type"),
                record.get("seebeck_coefficient"),
                record.get("electrical_conductivity"),
                record.get("thermal_conductivity"),
                record.get("power_factor"),
                record.get("zt_value"),
                record.get("temperature_k"),
                record.get("synthesis_method"),
                record.get("application"),
                record.get("confidence", "low"),
            ),
        )
        self.conn.commit()

    def insert_screening_result(self, result: dict):
        """Insert ML screening result."""
        self.conn.execute(
            """INSERT INTO screening_results
               (mxene_composition, composite_partner, structure_file,
                formation_energy, energy_above_hull, is_stable,
                predicted_seebeck, predicted_conductivity, predicted_kappa,
                predicted_zt, ml_model, relaxed_structure_file, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.get("mxene_composition"),
                result.get("composite_partner"),
                result.get("structure_file"),
                result.get("formation_energy"),
                result.get("energy_above_hull"),
                1 if result.get("is_stable") else 0,
                result.get("predicted_seebeck"),
                result.get("predicted_conductivity"),
                result.get("predicted_kappa"),
                result.get("predicted_zt"),
                result.get("ml_model"),
                result.get("relaxed_structure_file"),
                result.get("notes"),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------
    def query_by_composition(self, composition: str) -> list[dict]:
        """Find all TE records for a given MXene composition."""
        cur = self.conn.execute(
            "SELECT * FROM te_records WHERE mxene_composition = ?",
            (composition,),
        )
        return [dict(row) for row in cur.fetchall()]

    def query_by_partner(self, partner: str) -> list[dict]:
        """Find all TE records involving a specific composite partner."""
        cur = self.conn.execute(
            "SELECT * FROM te_records WHERE composite_partner = ?",
            (partner,),
        )
        return [dict(row) for row in cur.fetchall()]

    def query_top_performers(self, metric: str = "power_factor", limit: int = 20) -> list[dict]:
        """
        Get top-performing MXene composites by a given metric.

        VALID METRICS: seebeck_coefficient, electrical_conductivity,
                       thermal_conductivity, power_factor, zt_value
        """
        allowed = {
            "seebeck_coefficient", "electrical_conductivity",
            "thermal_conductivity", "power_factor", "zt_value",
        }
        if metric not in allowed:
            raise ValueError(f"Invalid metric. Choose from: {allowed}")

        cur = self.conn.execute(
            f"""SELECT te.*, p.title, p.year
                FROM te_records te
                JOIN papers p ON te.paper_id = p.paper_id
                WHERE te.{metric} IS NOT NULL
                ORDER BY te.{metric} DESC
                LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_unique_compositions(self) -> list[str]:
        """Get all unique MXene compositions in the database."""
        cur = self.conn.execute(
            "SELECT DISTINCT mxene_composition FROM te_records WHERE mxene_composition IS NOT NULL"
        )
        return [row[0] for row in cur.fetchall()]

    def get_unique_partners(self) -> list[str]:
        """Get all unique composite partners in the database."""
        cur = self.conn.execute(
            "SELECT DISTINCT composite_partner FROM te_records WHERE composite_partner IS NOT NULL"
        )
        return [row[0] for row in cur.fetchall()]

    def gap_analysis(self) -> dict:
        """
        CORE NOVELTY: Identify unexplored MXene-composite combinations.

        ALGORITHM:
            1. Get all unique compositions and partners from literature
            2. Build the full cross-product matrix (every possible combination)
            3. Query which combinations actually have data
            4. The DIFFERENCE = unexplored combinations = discovery targets

        RETURNS:
            {
                "explored": [(comp, partner), ...],
                "unexplored": [(comp, partner), ...],
                "exploration_rate": float,  # percentage explored
                "composition_coverage": {comp: num_partners_tested},
                "partner_coverage": {partner: num_compositions_tested},
            }

        WHY THIS MATTERS:
            Most researchers study Ti3C2Tx (because it's easiest to make).
            Other MXenes like V2CTx, Nb2CTx, Mo2TiC2Tx are barely explored
            for thermoelectrics. This analysis quantifies exactly where
            the blind spots are.
        """
        compositions = self.get_unique_compositions()
        partners = self.get_unique_partners()

        # Get existing combinations
        cur = self.conn.execute(
            """SELECT DISTINCT mxene_composition, composite_partner
               FROM te_records
               WHERE mxene_composition IS NOT NULL
               AND composite_partner IS NOT NULL"""
        )
        explored = set()
        for row in cur.fetchall():
            explored.add((row[0], row[1]))

        # Full cross product
        all_combinations = set()
        for c in compositions:
            for p in partners:
                all_combinations.add((c, p))

        unexplored = all_combinations - explored

        # Coverage analysis
        comp_coverage = {}
        for c in compositions:
            comp_coverage[c] = sum(1 for e in explored if e[0] == c)

        partner_coverage = {}
        for p in partners:
            partner_coverage[p] = sum(1 for e in explored if e[1] == p)

        total = len(all_combinations) if all_combinations else 1
        return {
            "explored": sorted(explored),
            "unexplored": sorted(unexplored),
            "exploration_rate": len(explored) / total * 100,
            "composition_coverage": comp_coverage,
            "partner_coverage": partner_coverage,
            "total_possible": len(all_combinations),
            "total_explored": len(explored),
            "total_unexplored": len(unexplored),
        }

    def get_summary_stats(self) -> dict:
        """Database-wide statistics."""
        stats = {}
        for table in ["papers", "te_records", "screening_results"]:
            cur = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cur.fetchone()[0]

        cur = self.conn.execute(
            "SELECT AVG(seebeck_coefficient), MAX(seebeck_coefficient) FROM te_records WHERE seebeck_coefficient IS NOT NULL"
        )
        row = cur.fetchone()
        stats["avg_seebeck"] = row[0]
        stats["max_seebeck"] = row[1]

        cur = self.conn.execute(
            "SELECT AVG(zt_value), MAX(zt_value) FROM te_records WHERE zt_value IS NOT NULL"
        )
        row = cur.fetchone()
        stats["avg_zt"] = row[0]
        stats["max_zt"] = row[1]

        return stats

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
