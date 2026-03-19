"""
paper_fetcher.py - Fetch MXene thermoelectric papers from Semantic Scholar API

ALGORITHM:
    1. Send search queries to Semantic Scholar Graph API (free, no key needed)
    2. For each query, paginate through results collecting paper metadata
    3. Deduplicate by paper ID (same paper may appear in multiple queries)
    4. Attempt to fetch full text / PDF links for open-access papers
    5. Store raw results in JSON for downstream extraction

DATA STRUCTURES:
    - Paper: Pydantic model with title, abstract, authors, year, DOI, citations,
      venue, open_access_pdf URL, tldr (auto-summary from S2)
    - PaperStore: List[Paper] serialized to JSON lines for incremental storage

TECHNIQUES:
    - Rate limiting via time.sleep() to stay within free tier (100 req / 5 min)
    - Exponential backoff on HTTP 429 (rate limit exceeded)
    - Bulk endpoint for fetching paper details in batches of 500
    - Semantic Scholar's TLDR field gives us free abstractive summaries

API REFERENCE:
    https://api.semanticscholar.org/api-docs/graph
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from loguru import logger


@dataclass
class Paper:
    """Represents a single scientific paper with metadata."""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    year: Optional[int] = None
    authors: list[str] = field(default_factory=list)
    doi: Optional[str] = None
    venue: Optional[str] = None
    citation_count: int = 0
    reference_count: int = 0
    open_access_pdf: Optional[str] = None
    tldr: Optional[str] = None
    fields_of_study: list[str] = field(default_factory=list)
    url: Optional[str] = None


class PaperFetcher:
    """
    Fetches papers from Semantic Scholar API.

    HOW IT WORKS:
    - Uses the /paper/search endpoint for keyword queries
    - Each query returns up to 100 results per page (API limit)
    - We paginate to collect up to `max_per_query` papers per query
    - Papers are deduplicated by their Semantic Scholar paper_id
    - Results are saved incrementally to a JSONL file (one JSON per line)
      so we can resume if interrupted

    RATE LIMITING:
    - Free tier: 100 requests per 5 minutes
    - We enforce a delay of `1/rate_limit` seconds between requests
    - On HTTP 429, we do exponential backoff (wait 60s, then 120s, etc.)
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    # Fields we request from the API (controls what metadata comes back)
    # NOTE: 'doi' and 'url' are no longer standalone fields.
    #   DOI is inside 'externalIds' (e.g., externalIds.DOI)
    #   Paper URL can be constructed from paperId.
    PAPER_FIELDS = (
        "title,abstract,year,authors,externalIds,venue,citationCount,"
        "referenceCount,openAccessPdf,tldr,fieldsOfStudy"
    )

    def __init__(
        self,
        output_dir: str | Path = "data/papers",
        rate_limit: float = 3.0,
        max_per_query: int = 500,
        api_key: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # With API key: 1 req/sec is safe. Without: 5s minimum to avoid 429s.
        # The free tier aggressively throttles bursts, especially with large
        # result sets (limit=100). We use conservative delays to be safe.
        if api_key:
            self.delay = max(1.0, 1.0 / rate_limit)
            self.page_size = 100  # Full page size with API key
        else:
            self.delay = 5.0  # 5 seconds between requests for free tier
            self.page_size = 20  # Smaller pages = less data per request = fewer 429s
        self.api_key = api_key
        self.headers = {"x-api-key": api_key} if api_key else {}
        self.max_per_query = max_per_query
        self.seen_ids: set[str] = set()
        self.papers: list[Paper] = []

        # Load previously fetched papers to avoid re-fetching
        self._load_existing()

    def _load_existing(self):
        """Load previously fetched papers from JSONL file."""
        jsonl_path = self.output_dir / "papers.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.seen_ids.add(data["paper_id"])
                    self.papers.append(Paper(**data))
            logger.info(f"Loaded {len(self.papers)} existing papers")

    def _request_with_backoff(self, url: str, params: dict, max_retries: int = 5):
        """
        Make HTTP GET request with exponential backoff.

        ALGORITHM:
            1. Send GET request
            2. If 200 OK -> return JSON
            3. If 429 Too Many Requests -> wait 60 * 2^retry seconds
            4. If other error -> log and return None
            5. Repeat up to max_retries times
        """
        for attempt in range(max_retries):
            time.sleep(self.delay)  # Rate limiting
            try:
                resp = requests.get(url, params=params, headers=self.headers, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = 60 * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait}s (attempt {attempt+1})")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
        return None

    def _parse_paper(self, raw: dict) -> Optional[Paper]:
        """
        Parse raw API response into Paper dataclass.

        Handles missing/null fields gracefully - the API doesn't guarantee
        all fields are present even when requested.
        """
        paper_id = raw.get("paperId")
        if not paper_id or paper_id in self.seen_ids:
            return None

        authors = []
        for a in (raw.get("authors") or []):
            if a.get("name"):
                authors.append(a["name"])

        pdf_url = None
        oap = raw.get("openAccessPdf")
        if oap and isinstance(oap, dict):
            pdf_url = oap.get("url") or None  # empty string → None

        tldr_text = None
        tldr = raw.get("tldr")
        if tldr and isinstance(tldr, dict):
            tldr_text = tldr.get("text")

        # DOI is now inside externalIds, not a top-level field
        doi = None
        ext_ids = raw.get("externalIds")
        if ext_ids and isinstance(ext_ids, dict):
            doi = ext_ids.get("DOI")

        # Construct paper URL from paperId
        paper_url = f"https://www.semanticscholar.org/paper/{paper_id}"

        return Paper(
            paper_id=paper_id,
            title=raw.get("title", ""),
            abstract=raw.get("abstract"),
            year=raw.get("year"),
            authors=authors,
            doi=doi,
            venue=raw.get("venue"),
            citation_count=raw.get("citationCount", 0),
            reference_count=raw.get("referenceCount", 0),
            open_access_pdf=pdf_url,
            tldr=tldr_text,
            fields_of_study=raw.get("fieldsOfStudy") or [],
            url=paper_url,
        )

    def search(self, query: str) -> list[Paper]:
        """
        Search for papers matching a query string.

        ALGORITHM:
            1. Send search query with offset=0, limit=page_size (20 free / 100 with key)
            2. Parse each result into Paper, skip duplicates
            3. Increment offset by page_size, repeat until:
               - No more results (total exhausted)
               - Reached max_per_query limit
            4. Return list of new (non-duplicate) papers

        PAGINATION:
            Semantic Scholar returns {total, offset, data[]} where:
            - total = total matching papers
            - offset = current position
            - data = list of paper objects (max 100 per page, we use 20 on free tier)
        """
        logger.info(f"Searching: '{query}'")
        new_papers = []
        offset = 0
        limit = self.page_size  # 100 with API key, 20 without (free tier safe)

        while offset < self.max_per_query:
            url = f"{self.BASE_URL}/paper/search"
            params = {
                "query": query,
                "offset": offset,
                "limit": limit,
                "fields": self.PAPER_FIELDS,
            }

            result = self._request_with_backoff(url, params)
            if not result or "data" not in result:
                break

            data = result["data"]
            if not data:
                break

            for raw in data:
                paper = self._parse_paper(raw)
                if paper:
                    self.seen_ids.add(paper.paper_id)
                    self.papers.append(paper)
                    new_papers.append(paper)

            total = result.get("total", 0)
            offset += limit
            if offset >= total:
                break

            logger.info(f"  Fetched {offset}/{min(total, self.max_per_query)} for '{query}'")

        logger.info(f"  Found {len(new_papers)} new papers for '{query}'")
        return new_papers

    def fetch_all(self, queries: list[str]) -> list[Paper]:
        """
        Run multiple search queries and collect all unique papers.

        ALGORITHM:
            1. For each query in the list, call self.search()
            2. Deduplication happens automatically via self.seen_ids
            3. After all queries, save results to JSONL file
            4. Also save a summary CSV for quick inspection
        """
        all_new = []
        for i, query in enumerate(queries):
            logger.info(f"Query {i+1}/{len(queries)}: '{query}'")
            new = self.search(query)
            all_new.extend(new)
            logger.info(f"  Total unique papers so far: {len(self.papers)}")
            # Wait between queries to avoid rate limiting across searches
            if i < len(queries) - 1:
                time.sleep(5)

        self._save()
        logger.info(f"Fetching complete. Total: {len(self.papers)}, New: {len(all_new)}")
        return all_new

    def _save(self):
        """
        Save all papers to disk in two formats:

        1. JSONL (papers.jsonl) - one JSON object per line
           Used for: machine reading, incremental appending
           Why JSONL: can append without loading entire file, handles large datasets

        2. CSV (papers_summary.csv) - tabular summary
           Used for: quick human inspection in Excel/pandas
        """
        # Save JSONL
        jsonl_path = self.output_dir / "papers.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for p in self.papers:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

        # Save CSV summary
        csv_path = self.output_dir / "papers_summary.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("paper_id,year,title,venue,citations,has_pdf,doi\n")
            for p in self.papers:
                title = p.title.replace('"', '""')
                venue = (p.venue or "").replace('"', '""')
                has_pdf = "yes" if p.open_access_pdf else "no"
                f.write(
                    f'"{p.paper_id}",{p.year or ""},"{title}","{venue}",'
                    f'{p.citation_count},{has_pdf},"{p.doi or ""}"\n'
                )

        logger.info(f"Saved {len(self.papers)} papers to {jsonl_path}")

    def get_stats(self) -> dict:
        """Return summary statistics about the paper collection."""
        years = [p.year for p in self.papers if p.year]
        with_abstract = sum(1 for p in self.papers if p.abstract)
        with_pdf = sum(1 for p in self.papers if p.open_access_pdf)
        return {
            "total_papers": len(self.papers),
            "with_abstract": with_abstract,
            "with_pdf": with_pdf,
            "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
            "total_citations": sum(p.citation_count for p in self.papers),
        }


# ---------------------------------------------------------------------------
# CLI entry point - run this file directly to fetch papers
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import yaml

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # API key is optional - if present, we get higher rate limits (1 req/sec)
    api_key = config["semantic_scholar"].get("api_key")
    if api_key:
        logger.info("Using Semantic Scholar API key (1 req/sec rate limit)")
    else:
        logger.info("No API key - using free tier (5s delay, 20 results/page)")

    fetcher = PaperFetcher(
        output_dir=Path(config["project"]["base_dir"]) / "data" / "papers",
        rate_limit=config["semantic_scholar"]["rate_limit_per_second"],
        max_per_query=config["semantic_scholar"]["max_papers"],
        api_key=api_key,
    )

    papers = fetcher.fetch_all(config["semantic_scholar"]["search_queries"])
    stats = fetcher.get_stats()
    print(f"\n--- Fetch Complete ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
