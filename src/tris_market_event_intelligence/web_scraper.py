"""Optional, polite HTML news discovery for local personal research.

Scraping is disabled unless explicitly run. No login, anti-bot bypass,
JavaScript execution, cross-site redirects, or API credentials required.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from tris_market_event_intelligence.collector import Article, NewsStore, utc_now
from tris_market_event_intelligence.events import clean_text

USER_AGENT = "TrisMarketEventIntelligenceBot/0.1 (personal local news research)"
MAX_HTML_BYTES = 1_500_000
MAX_CANDIDATES_PER_PORTAL = 150
BODY_LIMIT = 12000

# These are public landing-page *candidates*, not a claim of crawler permission.
# Robots rules are checked at run time for both index and individual articles.
PORTALS = {
    "naver": (
        "https://news.naver.com/section/101",
        frozenset({"news.naver.com"}),
    ),
    "daum": (
        "https://news.daum.net/economic",
        frozenset({"news.daum.net", "v.daum.net"}),
    ),
    "nate": (
        "https://news.nate.com/section?mid=n0200",
        frozenset({"news.nate.com"}),
    ),
}

NAVER_ARTICLE = re.compile(r"^/mnews/article/\d+/\d+|^/article/\d+/\d+")
DAUM_ARTICLE = re.compile(r"^/v/\d{8,}")
NATE_ARTICLE = re.compile(r"^/view/\d+")
ARTICLE_PATHS = {
    "naver": NAVER_ARTICLE, "daum": DAUM_ARTICLE, "nate": NATE_ARTICLE,
}


def _allowed_url(url: str, hosts: frozenset[str]) -> bool:
    try:
        parts = urlsplit(url)
        return (
            parts.scheme == "https" and (parts.hostname or "").lower() in hosts
            and not parts.username and not parts.password and not parts.fragment
            and parts.port is None and len(url) <= 2048
        )
    except ValueError:
        return False


class _NewsHTML(HTMLParser):
    """Small HTML parser: page links, metadata and marked article-body regions."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.metadata: dict[str, str] = {}
        self.heading: list[str] = []
        self.body: list[str] = []
        self._anchor: str | None = None
        self._anchor_text: list[str] = []
        self._body_depth = 0
        self._heading_depth = 0
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        if self._ignored_depth:
            return
        if tag == "meta":
            key = attrs.get("property") or attrs.get("name")
            value = attrs.get("content")
            if key and value and key not in self.metadata:
                self.metadata[key] = value
        if tag == "a":
            self._anchor = attrs.get("href")
            self._anchor_text = []
        if tag == "h1":
            self._heading_depth += 1
        if self._body_depth:
            self._body_depth += 1
        elif tag in {"article", "div", "section"}:
            ident = attrs.get("id", "")
            classes = attrs.get("class", "")
            if (
                tag == "article" or ident in {
                    "dic_area", "harmonyContainer", "articleCont",
                    "articleBodyContents", "article_view",
                } or "article_view" in classes.split()
            ):
                self._body_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "a" and self._anchor is not None:
            title = clean_text(" ".join(self._anchor_text))
            if title:
                self.links.append((self._anchor, title))
            self._anchor, self._anchor_text = None, []
        if tag == "h1" and self._heading_depth:
            self._heading_depth -= 1
        if self._body_depth:
            self._body_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth or not data.strip():
            return
        if self._anchor is not None:
            self._anchor_text.append(data)
        if self._heading_depth:
            self.heading.append(data)
        if self._body_depth:
            self.body.append(data)


def _parse_html(payload: bytes) -> _NewsHTML:
    if len(payload) > MAX_HTML_BYTES:
        raise ValueError("HTML payload exceeds 1.5 MB")
    parser = _NewsHTML()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return parser


def article_links(portal: str, page_url: str, payload: bytes) -> list[str]:
    parser = _parse_html(payload)
    hosts = PORTALS[portal][1]
    found: list[str] = []
    seen: set[str] = set()
    for href, _ in parser.links:
        url = urljoin(page_url, href)
        if not _allowed_url(url, hosts):
            continue
        parts = urlsplit(url)
        if not ARTICLE_PATHS[portal].match(parts.path):
            continue
        # Dedupe on canonical path, without tracking query parameters.
        canonical = f"https://{parts.hostname}{parts.path}"
        if canonical not in seen:
            found.append(canonical)
            seen.add(canonical)
        if len(found) >= MAX_CANDIDATES_PER_PORTAL:
            break
    return found


def parse_article(portal: str, url: str, payload: bytes, detected: str) -> Article | None:
    page = _parse_html(payload)
    title = clean_text(
        page.metadata.get("og:title")
        or page.metadata.get("twitter:title")
        or " ".join(page.heading)
    )
    content = clean_text(" ".join(page.body))[:BODY_LIMIT]
    if not title or not content or len(content) < 60:
        return None
    published = (
        page.metadata.get("article:published_time")
        or page.metadata.get("datePublished")
    )
    # Do not fabricate a publication timestamp from scraping time.
    from tris_market_event_intelligence.collector import _parse_date

    parsed = _parse_date(published) if published else None
    return Article(
        source=f"scrape_{portal}", external_id=url, url=url,
        title=title, summary=content, published_at=parsed,
        detected_at=detected,
    )


@dataclass(frozen=True)
class PortalResult:
    status: str
    inserted: int
    discovered: int
    blocked: int
    errors: int


class WebScrapingCollector:
    def __init__(
        self, store: NewsStore, client: httpx.Client | None = None,
        *, minimum_delay: float = 3.0, max_articles: int = 10,
    ) -> None:
        if not 1 <= max_articles <= 30:
            raise ValueError("max_articles must be between 1 and 30 per portal")
        if minimum_delay < 1:
            raise ValueError("minimum_delay must be at least one second")
        self.store = store
        self.client = client
        self.minimum_delay = minimum_delay
        self.max_articles = max_articles
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last_request: float | None = None

    def _get(self, client: httpx.Client, url: str) -> httpx.Response:
        if self._last_request is not None:
            delay = self.minimum_delay - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
        self._last_request = time.monotonic()
        result = client.get(url, headers={"User-Agent": USER_AGENT})
        # Redirects are intentionally not followed: reject cross-origin redirects.
        result.raise_for_status()
        if result.status_code in {301, 302, 303, 307, 308}:
            raise ValueError("Unexpected redirect; not following automatically")
        if len(result.content) > MAX_HTML_BYTES:
            raise ValueError("HTML payload exceeds 1.5 MB")
        return result

    def _permitted(self, client: httpx.Client, url: str) -> bool:
        host = urlsplit(url).hostname
        if host is None:
            return False
        if host not in self._robots:
            try:
                response = self._get(client, f"https://{host}/robots.txt")
                rules = RobotFileParser()
                rules.parse(response.text.splitlines())
                self._robots[host] = rules
            except httpx.HTTPStatusError as exc:
                # A missing robots.txt may mean no robots rules exist;
                # treat 401/403/429/5xx and all other statuses as blocked.
                if exc.response.status_code in {404, 410}:
                    self._robots[host] = None
                else:
                    return False
            except (httpx.HTTPError, ValueError):
                return False
        rules = self._robots[host]
        return rules is None or rules.can_fetch(USER_AGENT, url)

    def collect_once(self, portals: tuple[str, ...] = ("naver", "daum", "nate")) -> dict[str, PortalResult]:
        unknown = set(portals).difference(PORTALS)
        if unknown:
            raise ValueError(f"Unknown portal: {', '.join(sorted(unknown))}")
        self.store.initialize()
        owned = self.client is None
        client = self.client or httpx.Client(
            timeout=15, trust_env=False, follow_redirects=False,
        )
        result: dict[str, PortalResult] = {}
        try:
            for portal in portals:
                landing, hosts = PORTALS[portal]
                blocked = errors = inserted = discovered = 0
                if not self._permitted(client, landing):
                    result[portal] = PortalResult("blocked_by_robots", 0, 0, 1, 0)
                    continue
                try:
                    response = self._get(client, landing)
                    links = article_links(portal, landing, response.content)
                except (httpx.HTTPError, ValueError) as exc:
                    self.store.record_error(f"scrape_{portal}", type(exc).__name__, utc_now())
                    result[portal] = PortalResult("error", 0, 0, 0, 1)
                    continue
                discovered = len(links)
                for url in links[:self.max_articles]:
                    if not _allowed_url(url, hosts) or not self._permitted(client, url):
                        blocked += 1
                        continue
                    try:
                        response = self._get(client, url)
                        if "html" not in response.headers.get("content-type", "").lower():
                            raise ValueError("Unexpected non-HTML content")
                        now = utc_now()
                        article = parse_article(portal, url, response.content, now)
                        if article is None:
                            raise ValueError("Article body or title not found")
                        inserted += self.store.save(
                            article.source, url, response.content, [article],
                            now, None, None,
                        )
                    except (httpx.HTTPError, ValueError) as exc:
                        self.store.record_error(
                            f"scrape_{portal}", type(exc).__name__, utc_now()
                        )
                        errors += 1
                status = "ok" if errors == blocked == 0 else "partial"
                result[portal] = PortalResult(status, inserted, discovered, blocked, errors)
        finally:
            if owned:
                client.close()
        return result
