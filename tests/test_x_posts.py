import threading
import time
from unittest.mock import MagicMock, patch

import httpx

from health_agent.config import Settings
from health_agent.x_posts import (
    MAX_POST_CHARS,
    Post,
    attach_posts,
    fetch_post,
    select_matches,
    syndication_token,
)

SETTINGS = Settings(
    voyage_api_key="test-voyage-key",
    anthropic_api_key="test-anthropic-key",
    xai_api_key="test-xai-key",
)


def test_syndication_token_matches_react_tweet():
    # Expected values from react-tweet's getToken, run in Node.
    assert syndication_token("2112690385533522344") == "54d7nit4kh"
    assert syndication_token("1790552785290682735") == "4c96qyvwwt"
    assert syndication_token("20") == "6dq1a2xwd93"


def test_select_matches_keeps_posts_scoring_at_least_half():
    assert select_matches([(4, 0.97), (1, 0.65), (2, 0.43)]) == [4, 1]
    assert select_matches([(2, 0.50), (0, 0.49)]) == [2]
    assert select_matches([(0, 0.46), (1, 0.42)]) == []
    assert select_matches([(0, 0.95), (1, 0.94), (2, 0.93), (3, 0.92)]) == [0, 1, 2]
    assert select_matches([]) == []


def test_fetch_post_uses_long_text_and_canonical_url():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["id"] == "111"
        return httpx.Response(
            200,
            json={
                "text": "truncated…",
                "note_tweet": {"text": "the full long-form post"},
                "user": {"screen_name": "foundmyfitness"},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        post = fetch_post(client, "https://x.com/i/status/111")

    assert post == Post(
        url="https://x.com/foundmyfitness/status/111",
        handle="foundmyfitness",
        text="the full long-form post",
    )


def test_fetch_post_skips_deleted_and_failed_posts():
    responses = iter(
        [
            httpx.Response(200, json={"__typename": "TweetTombstone"}),
            httpx.Response(404, json={}),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(lambda r: next(responses))) as client:
        assert fetch_post(client, "https://x.com/i/status/1") is None
        assert fetch_post(client, "https://x.com/i/status/2") is None
        assert fetch_post(client, "https://example.com/not-a-post") is None


def test_fetch_post_includes_the_post_replied_to_and_the_quoted_post():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "text": "Here's how.",
                "user": {"screen_name": "grimhood"},
                "parent": {"text": "Removing pesticides", "user": {"screen_name": "grimhood"}},
                "quoted_tweet": {"text": "Residue study", "user": {"screen_name": "lab"}},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        post = fetch_post(client, "https://x.com/i/status/111")

    assert post is not None
    assert post.text == (
        "Replying to @grimhood: Removing pesticides\n\n"
        "Here's how.\n\n"
        "Quoting @lab: Residue study"
    )


def test_fetch_post_caps_long_posts():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "x" * 5000, "user": {"screen_name": "a"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        post = fetch_post(client, "https://x.com/i/status/111")

    assert post is not None
    assert len(post.text) == MAX_POST_CHARS


def _finding(claim, urls=None):
    return {"claim": claim, "relevance": "why", "source_urls": urls or []}


POSTS = [
    Post("https://x.com/a/status/1", "a", "Sleep loss lowers testosterone 10-15%."),
    Post("https://x.com/b/status/2", "b", "GABA stops cortisol release."),
]


def test_attach_posts_gives_findings_their_posts_text():
    findings = [
        _finding("One week of short sleep cuts testosterone 10-15%."),
        _finding("Already linked by Grok", ["https://x.com/b/status/2"]),
        _finding("Something no post supports."),
    ]
    scores = {
        findings[0]["claim"]: [(0, 0.96), (1, 0.40)],
        findings[2]["claim"]: [(1, 0.45)],
    }

    with (
        patch("health_agent.x_posts.fetch_posts", return_value=POSTS) as fetch,
        patch(
            "health_agent.x_posts._rank_posts",
            side_effect=lambda claim, posts, settings: scores[claim],
        ),
    ):
        result = attach_posts(
            findings, ["https://x.com/i/status/1", "https://x.com/i/status/2"], SETTINGS
        )

    # Grok's own source URLs are fetched once, deduplicated by post ID.
    fetch.assert_called_once_with(["https://x.com/i/status/1", "https://x.com/i/status/2"])
    assert result[0]["source_urls"] == ["https://x.com/a/status/1"]
    assert result[0]["matched"] is True
    assert result[0]["posts"] == [{"handle": "a", "text": POSTS[0].text}]
    assert result[1] == {**findings[1], "posts": [{"handle": "b", "text": POSTS[1].text}]}
    assert result[2] == findings[2]


def test_attach_posts_keeps_grok_linked_posts_when_matching_fails():
    findings = [_finding("linked", ["https://x.com/b/status/2"]), _finding("unlinked")]
    with (
        patch("health_agent.x_posts.fetch_posts", return_value=POSTS),
        patch("health_agent.x_posts._rank_posts", side_effect=RuntimeError("voyage down")),
    ):
        result = attach_posts(findings, ["https://x.com/i/status/1"], SETTINGS)

    assert result[0]["posts"] == [{"handle": "b", "text": POSTS[1].text}]
    assert result[1] == findings[1]


def test_attach_posts_skips_work_when_there_are_no_posts():
    fetch = MagicMock()
    with patch("health_agent.x_posts.fetch_posts", fetch):
        assert attach_posts([_finding("claim")], [], SETTINGS) == [_finding("claim")]
        assert attach_posts([], ["https://x.com/i/status/1"], SETTINGS) == []
    fetch.assert_not_called()


def test_attach_posts_returns_findings_unchanged_on_failure():
    findings = [_finding("claim", ["https://x.com/a/status/1"])]
    with patch("health_agent.x_posts.fetch_posts", side_effect=RuntimeError("x down")):
        assert attach_posts(findings, ["https://x.com/i/status/1"], SETTINGS) == findings


def test_attach_posts_gives_up_after_its_time_budget():
    """A stalled X fetch costs the chat POSTS_BUDGET_S, not the fetch's own timeouts."""
    findings = [_finding("claim")]
    release = threading.Event()

    def stalled_fetch(urls):
        release.wait(timeout=5)
        return []

    with (
        patch("health_agent.x_posts.fetch_posts", side_effect=stalled_fetch),
        patch("health_agent.x_posts.POSTS_BUDGET_S", 0.05),
    ):
        started = time.monotonic()
        result = attach_posts(findings, ["https://x.com/i/status/1"], SETTINGS)
        elapsed = time.monotonic() - started
        release.set()

    assert result == findings
    assert elapsed < 1
