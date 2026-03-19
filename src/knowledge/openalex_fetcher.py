"""
openalex_fetcher.py - Fetch MXene thermoelectric papers from OpenAlex API

WHY OPENALEX:
    OpenAlex is a free, open scholarly database (successor to Microsoft Academic).
    Unlike Semantic Scholar, it has no aggressive rate limiting for unauthenticated
    users — just include an email in User-Agent for the "polite pool" (10 req/sec).
    We use this as a reliable alternative when Semantic Scholar's free tier
    returns persistent 429 errors.

ALGORITHM:
    1. For each search query, call OpenAlex /works endpoint with pagination
    2. Parse results into the same Paper dataclass as paper_fetcher.py
    3. Deduplicate by DOI (cross-source) and OpenAlex ID
    4. Reconstruct abstracts from OpenAlex's inverted index format
    5. Save results in identical JSONL format for downstream compatibility

API REFERENCE:
    https://docs.openalex.org/api-entities/works

DATA STRUCTURES:
    - Paper: Same dataclass from paper_fetcher.py (reused for pipeline compatibility)
    - InvertedIndex: OpenAlex stores abstracts as {word: [position_list]} dicts.
      We reconstruct plaintext by sorting positions and joining words.

TECHNIQUES:
    - "Polite pool" access: include mailto: in User-Agent for 10 req/sec
    - Cursor-based pagination (faster than offset for large result sets)
    - Inverted index reconstruction for abstracts
    - Cross-source deduplication via DOI matching
"""

import json
import time
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

# Reuse the same Paper dataclass from paper_fetcher
from src.knowledge.paper_fetcher import Paper


class OpenAlexFetcher:
    """
    Fetches papers from OpenAlex API — a free, open scholarly database.

    DESIGN:
    - Outputs the exact same Paper dataclass and JSONL format as PaperFetcher
    - This means the rest of the pipeline (extractor, database, gap analyzer)
      works identically regardless of which data source was used
    - OpenAlex has broader coverage than Semantic Scholar for some fields
    - Polite pool (10 req/sec) is more reliable than S2's free tier

    RATE LIMITING:
    - Polite pool (with email in User-Agent): 10 requests per second
    - We use 1-second delay to be safe and courteous
    """

    BASE_URL = "https://api.openalex.org"

    # Fields we select from OpenAlex (reduces response size and improves speed)
    SELECT_FIELDS = (
        "id,title,publication_year,doi,cited_by_count,authorships,"
        "primary_location,abstract_inverted_index,concepts,open_access,"
        "referenced_works_count"
    )

    def __init__(
        self,
        output_dir: str | Path = "data/papers",
        email: str = "mxdiscovery@research.org",
        max_per_query: int = 200,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.max_per_query = max_per_query
        self.delay = 1.0  # 1 second between requests (polite pool allows 10/sec)
        self.headers = {"User-Agent": f"mailto:{email}"}

        # Track seen papers to deduplicate
        self.seen_ids: set[str] = set()  # OpenAlex IDs
        self.seen_dois: set[str] = set()  # DOIs for cross-source dedup
        self.papers: list[Paper] = []

        # Load previously fetched papers
        self._load_existing()

    def _load_existing(self):
        """Load previously fetched papers from JSONL file to avoid re-fetching."""
        jsonl_path = self.output_dir / "papers.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        self.seen_ids.add(data["paper_id"])
                        if data.get("doi"):
                            self.seen_dois.add(data["doi"].lower())
                        self.papers.append(Paper(**data))
            logger.info(f"Loaded {len(self.papers)} existing papers")

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
        """
        Reconstruct plaintext abstract from OpenAlex's inverted index format.

        ALGORITHM:
            OpenAlex stores abstracts as: {"word1": [0, 5], "word2": [1, 3], ...}
            where values are position indices. To reconstruct:
            1. Invert the mapping: position -> word
            2. Sort by position
            3. Join with spaces

        WHY INVERTED INDEX:
            OpenAlex uses this format to save storage (common words like "the"
            are stored once with multiple positions instead of repeated).
        """
        if not inverted_index:
            return None

        positions = {}
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions[pos] = word

        if not positions:
            return None

        abstract = " ".join(positions[i] for i in sorted(positions))
        return abstract if len(abstract) > 20 else None  # Skip trivially short abstracts

    def _parse_work(self, work: dict) -> Optional[Paper]:
        """
        Parse an OpenAlex work into our Paper dataclass.

        FIELD MAPPING (OpenAlex -> Paper):
            id -> paper_id (stripped of URL prefix)
            title -> title
            abstract_inverted_index -> abstract (reconstructed)
            publication_year -> year
            authorships[].author.display_name -> authors
            doi -> doi (stripped of URL prefix)
            primary_location.source.display_name -> venue
            cited_by_count -> citation_count
            referenced_works_count -> reference_count
            open_access.oa_url -> open_access_pdf
            concepts[0..2].display_name -> fields_of_study
        """
        openalex_id = work.get("id", "")
        # Strip URL prefix: "https://openalex.org/W12345" -> "W12345"
        paper_id = openalex_id.replace("https://openalex.org/", "")

        if not paper_id or paper_id in self.seen_ids:
            return None

        # Extract DOI (comes as URL like "https://doi.org/10.1234/...")
        doi_url = work.get("doi")
        doi = None
        if doi_url:
            doi = doi_url.replace("https://doi.org/", "")
            # Cross-source deduplication: skip if we already have this DOI
            if doi.lower() in self.seen_dois:
                return None

        # Authors
        authors = []
        for authorship in (work.get("authorships") or []):
            author = authorship.get("author", {})
            name = author.get("display_name")
            if name:
                authors.append(name)

        # Venue from primary location
        venue = None
        primary_loc = work.get("primary_location")
        if primary_loc and isinstance(primary_loc, dict):
            source = primary_loc.get("source")
            if source and isinstance(source, dict):
                venue = source.get("display_name")

        # Open access PDF URL
        pdf_url = None
        oa = work.get("open_access")
        if oa and isinstance(oa, dict):
            pdf_url = oa.get("oa_url")

        # Fields of study from concepts (top 3 by score)
        fields = []
        concepts = work.get("concepts") or []
        for concept in concepts[:5]:  # Take top 5 concepts
            name = concept.get("display_name")
            if name:
                fields.append(name)

        # Reconstruct abstract from inverted index
        abstract = self._reconstruct_abstract(work.get("abstract_inverted_index"))

        # Construct Semantic Scholar-style URL (for consistency)
        # We use OpenAlex URL since we don't have S2 paper ID
        paper_url = f"https://openalex.org/{paper_id}"

        return Paper(
            paper_id=paper_id,
            title=work.get("title", ""),
            abstract=abstract,
            year=work.get("publication_year"),
            authors=authors,
            doi=doi,
            venue=venue,
            citation_count=work.get("cited_by_count", 0),
            reference_count=work.get("referenced_works_count", 0),
            open_access_pdf=pdf_url,
            tldr=None,  # OpenAlex doesn't have auto-summaries
            fields_of_study=fields,
            url=paper_url,
        )

    def search(self, query: str) -> list[Paper]:
        """
        Search OpenAlex for papers matching a query.

        ALGORITHM:
            1. Send search request with cursor-based pagination
            2. Parse each work into Paper dataclass
            3. Continue until max_per_query reached or no more results
            4. OpenAlex cursor pagination: use cursor="*" for first page,
               then use meta.next_cursor for subsequent pages

        PAGINATION:
            OpenAlex supports two modes:
            - offset-based: slower, limited to 10,000 results
            - cursor-based: faster, unlimited results
            We use cursor-based for reliability.
        """
        logger.info(f"Searching OpenAlex: '{query}'")
        new_papers = []
        cursor = "*"  # Start cursor
        collected = 0
        per_page = 50  # OpenAlex max is 200, we use 50 for steady progress

        while collected < self.max_per_query:
            time.sleep(self.delay)

            params = {
                "search": query,
                "per_page": min(per_page, self.max_per_query - collected),
                "select": self.SELECT_FIELDS,
                "cursor": cursor,
                "sort": "cited_by_count:desc",  # Most cited first = highest quality
            }

            try:
                resp = requests.get(
                    f"{self.BASE_URL}/works",
                    params=params,
                    headers=self.headers,
                    timeout=30,
                )

                if resp.status_code != 200:
                    logger.error(f"OpenAlex HTTP {resp.status_code}: {resp.text[:200]}")
                    break

                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                for work in results:
                    paper = self._parse_work(work)
                    if paper:
                        self.seen_ids.add(paper.paper_id)
                        if paper.doi:
                            self.seen_dois.add(paper.doi.lower())
                        self.papers.append(paper)
                        new_papers.append(paper)
                        collected += 1

                # Get next cursor for pagination
                meta = data.get("meta", {})
                next_cursor = meta.get("next_cursor")
                total = meta.get("count", 0)

                if not next_cursor or collected >= self.max_per_query:
                    break
                cursor = next_cursor

                logger.info(
                    f"  Fetched {collected}/{min(total, self.max_per_query)} "
                    f"for '{query}' ({len(new_papers)} new)"
                )

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                time.sleep(5)
                break

        logger.info(f"  Found {len(new_papers)} new papers for '{query}'")
        return new_papers

    def fetch_all(self, queries: list[str]) -> list[Paper]:
        """
        Run all search queries and collect unique papers.

        Same interface as PaperFetcher.fetch_all() for drop-in compatibility.
        """
        all_new = []
        for i, query in enumerate(queries):
            logger.info(f"Query {i+1}/{len(queries)}: '{query}'")
            new = self.search(query)
            all_new.extend(new)
            logger.info(f"  Total unique papers so far: {len(self.papers)}")
            # Brief pause between queries
            if i < len(queries) - 1:
                time.sleep(2)

        self._save()
        logger.info(f"Fetching complete. Total: {len(self.papers)}, New: {len(all_new)}")
        return all_new

    def _save(self):
        """
        Save all papers to JSONL and summary CSV.

        Same format as PaperFetcher._save() for pipeline compatibility.
        """
        from dataclasses import asdict
        import csv

        jsonl_path = self.output_dir / "papers.jsonl"
        csv_path = self.output_dir / "papers_summary.csv"

        # JSONL: one JSON object per line (machine-readable, appendable)
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for paper in self.papers:
                f.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.papers)} papers to {jsonl_path}")

        # CSV summary: human-readable quick reference
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["title", "year", "doi", "citations", "venue", "has_abstract"])
            for p in self.papers:
                writer.writerow([
                    p.title[:100],
                    p.year,
                    p.doi or "",
                    p.citation_count,
                    p.venue or "",
                    "yes" if p.abstract else "no",
                ])
        logger.info(f"Saved summary to {csv_path}")

    def get_stats(self) -> dict:
        """Return summary statistics about fetched papers."""
        years = [p.year for p in self.papers if p.year]
        with_abstract = sum(1 for p in self.papers if p.abstract)
        with_doi = sum(1 for p in self.papers if p.doi)
        return {
            "total_papers": len(self.papers),
            "with_abstract": with_abstract,
            "without_abstract": len(self.papers) - with_abstract,
            "with_doi": with_doi,
            "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
            "total_citations": sum(p.citation_count for p in self.papers),
            "source": "OpenAlex",
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    fetcher = OpenAlexFetcher(
        output_dir=Path(config["project"]["base_dir"]) / "data" / "papers",
        max_per_query=config["semantic_scholar"]["max_papers"],
    )

    papers = fetcher.fetch_all(config["semantic_scholar"]["search_queries"])
    stats = fetcher.get_stats()
    print(f"\n--- OpenAlex Fetch Complete ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
