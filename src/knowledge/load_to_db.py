"""
load_to_db.py - Load fetched papers and extracted records into SQLite database

ALGORITHM:
    1. Read papers from data/papers/papers.jsonl
    2. Read extracted TE records from data/papers/extracted_records.jsonl
    3. Insert all papers into the papers table
    4. Insert all extracted records into the te_records table
    5. Run gap analysis and print summary
    6. Export gap analysis results for Stage 2 (gap_analyzer.py)

WHY SEPARATE SCRIPT:
    The paper fetcher and data extractor write JSONL files (flat, appendable).
    This script loads them into SQLite for relational queries. Keeping them
    separate means we can re-run extraction without re-loading everything,
    and we can inspect raw JSONL files without needing a database.
"""

import json
from pathlib import Path

from loguru import logger

from src.knowledge.database import MXeneDatabase


def load_papers(db: MXeneDatabase, papers_path: Path) -> int:
    """Load papers from JSONL into database."""
    count = 0
    with open(papers_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            paper = json.loads(line.strip())
            db.insert_paper(
                paper_id=paper["paper_id"],
                title=paper.get("title", ""),
                abstract=paper.get("abstract"),
                year=paper.get("year"),
                authors=json.dumps(paper.get("authors", [])),
                doi=paper.get("doi"),
                venue=paper.get("venue"),
                citations=paper.get("citation_count", 0),
                has_pdf=bool(paper.get("open_access_pdf")),
                tldr=paper.get("tldr"),
            )
            count += 1
    return count


def load_te_records(db: MXeneDatabase, records_path: Path) -> int:
    """Load extracted TE records from JSONL into database."""
    count = 0
    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line.strip())
            db.insert_te_record(record)
            count += 1
    return count


if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_dir = Path(config["project"]["base_dir"])
    papers_path = base_dir / "data" / "papers" / "papers.jsonl"
    records_path = base_dir / "data" / "papers" / "extracted_records.jsonl"
    db_path = base_dir / config["database"]["path"]

    db = MXeneDatabase(db_path)

    # Load papers
    if papers_path.exists():
        n = load_papers(db, papers_path)
        logger.info(f"Loaded {n} papers into database")
    else:
        logger.warning(f"No papers file at {papers_path}")

    # Load extracted records
    if records_path.exists():
        n = load_te_records(db, records_path)
        logger.info(f"Loaded {n} TE records into database")
    else:
        logger.warning(f"No extracted records at {records_path}")

    # Print summary
    stats = db.get_summary_stats()
    print("\n--- Database Summary ---")
    for k, v in stats.items():
        if isinstance(v, float) and v is not None:
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Run gap analysis
    gap = db.gap_analysis()
    print(f"\n--- Gap Analysis ---")
    print(f"  Unique compositions: {len(gap['composition_coverage'])}")
    print(f"  Unique partners: {len(gap['partner_coverage'])}")
    print(f"  Explored combinations: {gap['total_explored']}")
    print(f"  Unexplored combinations: {gap['total_unexplored']}")
    print(f"  Exploration rate: {gap['exploration_rate']:.1f}%")

    if gap['composition_coverage']:
        print(f"\n  Top explored compositions:")
        for comp, count in sorted(gap['composition_coverage'].items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {comp}: {count} partners tested")

    if gap['partner_coverage']:
        print(f"\n  Top explored partners:")
        for partner, count in sorted(gap['partner_coverage'].items(),
                                      key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {partner}: {count} compositions tested")

    db.close()
