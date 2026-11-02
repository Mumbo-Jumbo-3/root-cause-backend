"""Attach the X posts behind each Grok finding, with their text.

Grok's findings paraphrase posts, so later steps judge and cite the posts' own text
instead. We fetch each post the search read from X's public syndication API (the
same one the frontend's tweet embeds use). Grok rarely says which post supports
which finding, so findings without source URLs are matched by reranking the posts
against their claim. Close, high-scoring matches become that finding's sources,
flagged as `matched` so the UI doesn't present them as Grok's own citations.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
from langchain_core.documents import Document

from health_agent.config import Settings
from health_agent.rag.retriever import rerank_documents
from health_agent.trace import tweet_status_id

logger = logging.getLogger(__name__)

SYNDICATION_URL = "https://cdn.syndication.twimg.com/tweet-result"
FETCH_TIMEOUT_S = 4.0
MAX_WORKERS = 8
# Reranks share Voyage's rate limit with RAG; up to three search steps match at once.
RERANK_WORKERS = 3
# Fetching and matching sit on the chat's critical path (enriched RAG waits for
# trusted search), so they get one fixed budget; past it, findings go out without posts.
POSTS_BUDGET_S = 5.0
# Long-form posts run to 25k characters; the opening carries the point.
MAX_POST_CHARS = 2000

# Any post scoring at least this well counts as one of a finding's sources. Set
# liberally: true sources scored 0.71-0.97 on real runs, and posts on the same
# topic that don't state the claim itself land around 0.5-0.7.
MIN_SCORE = 0.5
MAX_POSTS_PER_FINDING = 3

_RADIX_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class Post:
    url: str
    handle: str
    text: str


def _js_radix_string(value: float, radix: int = 36) -> str:
    """`Number.prototype.toString(radix)` for a positive double (port of V8's algorithm)."""
    integer = math.floor(value)
    fraction = value - integer
    delta = max(0.5 * (math.nextafter(value, math.inf) - value), math.nextafter(0.0, 1.0))
    fraction_digits: list[int] = []
    if fraction >= delta:
        while True:
            fraction *= radix
            delta *= radix
            digit = int(fraction)
            fraction_digits.append(digit)
            fraction -= digit
            if (fraction > 0.5 or (fraction == 0.5 and digit & 1)) and fraction + delta > 1:
                # Round up, carrying into earlier digits and then the integer part.
                while True:
                    if not fraction_digits:
                        integer += 1
                        break
                    rounded = fraction_digits.pop() + 1
                    if rounded < radix:
                        fraction_digits.append(rounded)
                        break
                break
            if fraction < delta:
                break
    integer_digits = ""
    remaining = int(integer)
    while True:
        remaining, digit = divmod(remaining, radix)
        integer_digits = _RADIX_DIGITS[digit] + integer_digits
        if remaining == 0:
            break
    if not fraction_digits:
        return integer_digits
    return integer_digits + "." + "".join(_RADIX_DIGITS[d] for d in fraction_digits)


def syndication_token(status_id: str) -> str:
    """The `token` react-tweet sends with syndication requests."""
    token = _js_radix_string(int(status_id) / 1e15 * math.pi)
    return token.replace(".", "").replace("0", "")


def fetch_post(client: httpx.Client, url: str) -> Post | None:
    status_id = tweet_status_id(url)
    if not status_id:
        return None
    try:
        response = client.get(
            SYNDICATION_URL,
            params={"id": status_id, "lang": "en", "token": syndication_token(status_id)},
        )
        if response.status_code != 200:
            return None
        data = response.json()
    except (httpx.HTTPError, ValueError):
        logger.debug("syndication fetch failed for %s", url, exc_info=True)
        return None
    if not isinstance(data, dict) or data.get("__typename") == "TweetTombstone":
        return None
    text = _tweet_text(data)
    handle = _tweet_handle(data)
    if not text.strip() or not handle:
        return None
    # A reply or quote often says little without the post it answers or quotes.
    parent = data.get("parent") or {}
    quoted = data.get("quoted_tweet") or {}
    if _tweet_text(parent):
        text = f"Replying to @{_tweet_handle(parent)}: {_tweet_text(parent)}\n\n{text}"
    if _tweet_text(quoted):
        text = f"{text}\n\nQuoting @{_tweet_handle(quoted)}: {_tweet_text(quoted)}"
    return Post(
        url=f"https://x.com/{handle}/status/{status_id}",
        handle=handle,
        text=text[:MAX_POST_CHARS],
    )


def _tweet_text(tweet: dict) -> str:
    return (tweet.get("note_tweet") or {}).get("text") or tweet.get("text") or ""


def _tweet_handle(tweet: dict) -> str:
    return (tweet.get("user") or {}).get("screen_name") or ""


def fetch_posts(urls: list[str]) -> list[Post]:
    """Fetch posts in parallel, dropping failures and reposts of identical text."""
    with httpx.Client(timeout=FETCH_TIMEOUT_S) as client:
        with ThreadPoolExecutor(MAX_WORKERS) as pool:
            fetched = list(pool.map(lambda url: fetch_post(client, url), urls))
    posts: list[Post] = []
    seen_text: set[str] = set()
    for post in fetched:
        if post is None:
            continue
        key = " ".join(post.text.split()).lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        posts.append(post)
    return posts


def select_matches(scores: list[tuple[int, float]]) -> list[int]:
    """Indices of posts that support a finding, given (index, score) best-first."""
    return [index for index, score in scores if score >= MIN_SCORE][:MAX_POSTS_PER_FINDING]


def _rank_posts(claim: str, posts: list[Post], settings: Settings) -> list[tuple[int, float]]:
    documents = [
        Document(page_content=post.text, metadata={"post_index": index})
        for index, post in enumerate(posts)
    ]
    ranked = rerank_documents(claim, documents, settings)
    return [(doc.metadata["post_index"], doc.metadata["relevance_score"]) for doc in ranked]


def attach_posts(
    findings: list[dict[str, object]], post_urls: list[str], settings: Settings
) -> list[dict[str, object]]:
    """Give each finding `posts`: the handle and text of the posts it rests on.

    Findings without source URLs first get the read posts whose text supports them.
    Never raises and never takes longer than POSTS_BUDGET_S: if posts can't be fetched
    in time, findings come back unchanged.
    """
    urls: dict[str, str] = {}
    for url in [*post_urls, *(u for f in findings for u in f.get("source_urls", []))]:  # type: ignore[union-attr]
        if status_id := tweet_status_id(url):
            urls.setdefault(status_id, url)
    if not findings or not urls:
        return findings
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="x-posts")
    future = executor.submit(_attach_posts, findings, list(urls.values()), settings)
    # Don't wait on exit: work past the budget finishes in the background and is dropped.
    executor.shutdown(wait=False)
    try:
        return future.result(timeout=POSTS_BUDGET_S)
    except TimeoutError:
        logger.warning(
            "Fetching X posts took over %.0fs; findings go out without posts", POSTS_BUDGET_S
        )
    except Exception:
        logger.exception("Attaching X posts to findings failed; findings go out without posts")
    return findings


def _attach_posts(
    findings: list[dict[str, object]], urls: list[str], settings: Settings
) -> list[dict[str, object]]:
    posts = fetch_posts(urls)
    if not posts:
        return findings
    by_id = {tweet_status_id(post.url): post for post in posts}
    attached: list[dict[str, object]] = []
    for finding in _match_posts(findings, posts, settings):
        sources = [
            by_id[status_id]
            for url in finding.get("source_urls", [])  # type: ignore[union-attr]
            if (status_id := tweet_status_id(url)) in by_id
        ]
        if sources:
            finding = {
                **finding,
                "posts": [{"handle": post.handle, "text": post.text} for post in sources],
            }
        attached.append(finding)
    return attached


def _match_posts(
    findings: list[dict[str, object]], posts: list[Post], settings: Settings
) -> list[dict[str, object]]:
    """Give findings without source URLs the posts whose text supports them."""
    pending = [i for i, finding in enumerate(findings) if not finding.get("source_urls")]
    if not pending:
        return findings
    try:
        with ThreadPoolExecutor(min(len(pending), RERANK_WORKERS)) as pool:
            rankings = list(
                pool.map(
                    lambda i: _rank_posts(str(findings[i]["claim"]), posts, settings), pending
                )
            )
    except Exception:
        logger.exception("Matching X posts to findings failed; leaving them unlinked")
        return findings

    matched = list(findings)
    for i, scores in zip(pending, rankings):
        indices = select_matches(scores)
        if indices:
            matched[i] = {
                **findings[i],
                "source_urls": [posts[index].url for index in indices],
                "matched": True,
            }
    return matched
