from types import SimpleNamespace

import pytest

from health_agent import server


@pytest.mark.asyncio
async def test_lifespan_checks_connections_before_checkout(monkeypatch):
    pool_options = {}

    class FakePool:
        max_size = 20

        def __init__(self, **kwargs):
            pool_options.update(kwargs)

        @staticmethod
        async def check_connection(_connection):
            pass

        async def open(self):
            pass

        async def close(self):
            pass

    class FakeCheckpointer:
        def __init__(self, _pool):
            pass

        async def setup(self):
            pass

    monkeypatch.setattr(
        server,
        "get_settings",
        lambda: SimpleNamespace(database_url="postgresql://localhost/test"),
    )
    monkeypatch.setattr(server, "AsyncConnectionPool", FakePool)
    monkeypatch.setattr(server, "AsyncPostgresSaver", FakeCheckpointer)
    monkeypatch.setattr(server, "build_graph", lambda *_args, **_kwargs: object())

    app = SimpleNamespace(state=SimpleNamespace())
    async with server.lifespan(app):
        pass

    assert pool_options["check"] is FakePool.check_connection


def test_token_payloads_carry_block_index_and_citations():
    from health_agent.server import _token_payloads

    assert _token_payloads("plain") == [{"content": "plain"}]
    assert _token_payloads(
        [
            {"type": "text", "text": "Zinc", "index": 2},
            {
                "type": "text",
                "text": "",
                "index": 2,
                "citations": [{"document_index": 0, "cited_text": "Zinc helps."}],
            },
            {"type": "text", "text": "", "index": 3},
            {"type": "thinking", "thinking": "hmm", "index": 0},
        ]
    ) == [
        {"content": "Zinc", "block": 2},
        {"content": "", "block": 2, "citations": [{"document_index": 0, "cited_text": "Zinc helps."}]},
    ]
