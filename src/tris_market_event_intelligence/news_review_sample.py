"""Source-only, bounded representative NEWS review queue.

Declared story groups, exact text and canonical URLs yield *candidate* units,
not verified independent events. Lexical similarity is an unmerged warning.
No model scores, inferred gold, market data, or TEST access.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import _cohort, _instant
from tris_market_event_intelligence.news_near_duplicate_audit import _tokens
from tris_market_event_intelligence.story_index import canonical_url

MAX_ITEMS = 5000
MAX_SELECTED = 100
MIN_SHARED = 5
SIMILARITY = 0.78


class _Union:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a: int, b: int) -> None:
        left, right = self.find(a), self.find(b)
        self.parent[max(left, right)] = min(left, right)


def _candidate_units(articles: list[dict]) -> list[list[dict]]:
    union = _Union([row["articleId"] for row in articles])
    known: dict[tuple[str, str], int] = {}
    for row in articles:
        candidate_keys = [
            ("text", row["textSha256"]),
            ("url", canonical_url(row["sourceUrl"])),
        ]
        group = row.get("storyGroupId")
        if group is not None:
            if type(group) is not int or group < 1:
                raise ValueError("Invalid archived story group")
            candidate_keys.append(("declared-story", str(group)))
        for key in candidate_keys:
            if key in known:
                union.join(row["articleId"], known[key])
            else:
                known[key] = row["articleId"]
    buckets: dict[int, list[dict]] = defaultdict(list)
    for article in articles:
        buckets[union.find(article["articleId"])].append(article)
    return [sorted(items, key=lambda a: (
        _instant(a["firstObservedAt"]), a["articleId"],
    )) for _, items in sorted(buckets.items())]


def _lexical_warnings(articles: list[dict], units: list[list[dict]]) -> tuple[int, int]:
    """Count potential cross-unit near repeats; *never* merge them automatically.

    Bounded exact candidate pairs for up to 5000 items, no raw excerpts
    in public CLI stdout. Uses postings rather than all-pairs comparisons.
    """
    memberships = {a["articleId"]: i for i, unit in enumerate(units)
                   for a in unit}
    postings: dict[str, set[int]] = defaultdict(set)
    tokens = {}
    warnings = 0
    affected: set[int] = set()
    for row in sorted(articles, key=lambda x: (
        _instant(x["firstObservedAt"]), x["articleId"],
    )):
        aid = row["articleId"]
        words = _tokens(row["title"])
        candidates: Counter[int] = Counter()
        for word in words:
            candidates.update(postings[word])
        for other, shared in candidates.items():
            if shared < MIN_SHARED or memberships[other] == memberships[aid]:
                continue
            similarity = shared / len(words | tokens[other])
            if similarity >= SIMILARITY:
                warnings += 1
                affected.update((memberships[other], memberships[aid]))
        tokens[aid] = words
        for word in words:
            postings[word].add(aid)
    return warnings, len(affected)


def prepare_news_review_sample(*, cohort_path: Path,
                               max_selected: int = 48) -> dict[str, object]:
    """One locally stored, source-pinned packet; human review is still blank."""
    if type(max_selected) is not int or not 1 <= max_selected <= MAX_SELECTED:
        raise ValueError("Requested review sample must be 1..100")
    raw, cohort = _cohort(cohort_path)
    articles = cohort["items"]
    if len(articles) > MAX_ITEMS:
        raise ValueError("Source cohort exceeds sample bound")
    units = _candidate_units(articles)
    warnings, affected = _lexical_warnings(articles, units)
    raw_sha = hashlib.sha256(raw).hexdigest()

    # Sample one per candidate unit; balance across observed UTC day AND
    # source rather than falsely treating 446 same-day reports as independent.
    representatives = [unit[0] for unit in units]
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in representatives:
        day = _instant(row["firstObservedAt"]).date().isoformat()
        strata[(day, row["source"])].append(row)
    for rows in strata.values():
        rows.sort(key=lambda row: (
            hashlib.sha256((raw_sha + ":" + str(row["articleId"])).encode()).hexdigest(),
            row["articleId"],
        ))
    selected: list[dict] = []
    strata_keys = sorted(strata)
    while len(selected) < min(max_selected, len(representatives)):
        progressed = False
        for key in strata_keys:
            if strata[key] and len(selected) < max_selected:
                selected.append(strata[key].pop(0))
                progressed = True
        if not progressed:
            break
    by_id = {article["articleId"]: i for i, unit in enumerate(units)
             for article in unit}
    entries = []
    for row in selected:
        unit = units[by_id[row["articleId"]]]
        entries.append({
            "articleId": row["articleId"],
            "textSha256": row["textSha256"],
            "source": row["source"],
            "sourceUrl": row["sourceUrl"],
            "title": row["title"],
            "summary": row.get("summary"),
            "firstObservedAt": row["firstObservedAt"],
            "candidateUnitArticleIds": [a["articleId"] for a in unit],
            "candidateUnitIsVerifiedIndependentEvent": False,
            "importance": None, "potentialMagnitude": None,
            "sourceAlignment": None, "sameStoryConfirmedByHuman": None,
            "reviewer": None, "reviewedAt": None, "reason": None,
        })
    # A selection packet is separate from the complete human-review schema;
    # nothing here is silently merged into an accepted gold-label document.
    return {
        "schemaVersion": "news_human_review_candidate_sample_v1",
        "status": "CANDIDATE_SAMPLE_UNREVIEWED",
        "cohortSha256": raw_sha,
        "samplingPolicy": "utc-day-source-round-robin-sha-v1",
        "totalArticles": len(articles),
        "candidateUnitCount": len(units),
        "multiArticleCandidateUnits": sum(len(unit) > 1 for unit in units),
        "articlesInMultiArticleUnits": sum(
            len(unit) for unit in units if len(unit) > 1
        ),
        "exactOrDeclaredLinksNotHumanVerified": True,
        "crossUnitLexicalCandidatePairs": warnings,
        "candidateUnitsWithLexicalWarnings": affected,
        "lexicalWarningsNotConfirmedDuplicates": True,
        "observedDayCountsUTC": dict(sorted(Counter(
            _instant(a["firstObservedAt"]).date().isoformat()
            for a in articles
        ).items())),
        "selectedArticles": len(entries),
        "unselectedCandidateUnits": len(units) - len(entries),
        "items": entries,
        "modelCalls": 0,
        "humanGoldCreated": False,
        "trainingDatasetEligible": False,
        "testGoldScored": False,
        "modelImprovementValidated": False,
        "promotionAllowed": False,
        "stockPricesUsed": False,
        "productionDbModified": False,
    }
