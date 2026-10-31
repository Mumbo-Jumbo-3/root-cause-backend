from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from health_agent.trace import doc_id, doc_info, docs_detail, grok_cited_urls, post_urls


def test_grok_cited_urls_reads_responses_api_annotations():
    raw = AIMessage(
        content=[
            {"type": "reasoning", "summary": []},
            {
                "type": "text",
                "text": "{}",
                "annotations": [
                    {"type": "url_citation", "url": "https://x.com/i/status/1"},
                    {"type": "url_citation", "url": "https://x.com/someone/status/1"},
                    {"type": "url_citation", "url": "https://example.com/page"},
                ],
            },
        ]
    )

    urls = grok_cited_urls(raw)
    assert urls == [
        "https://x.com/i/status/1",
        "https://x.com/someone/status/1",
        "https://example.com/page",
    ]
    assert post_urls(urls) == ["https://x.com/i/status/1"]


def test_doc_info_drops_title_repeated_as_first_heading():
    doc = Document(
        page_content="Zinc Guide > Dosage\n\nTake it with food.",
        metadata={"title": "Zinc Guide", "author": "A", "header_path": "Zinc Guide > Dosage"},
    )

    info = doc_info(doc)
    assert info["section"] == ["Dosage"]
    assert info["text"] == "Take it with food."


def test_docs_detail_omits_text_already_sent_by_an_earlier_step():
    seen = Document(page_content="seen", metadata={"source": "a.md", "title": "A"})
    new = Document(page_content="new", metadata={"source": "b.md", "title": "B"})

    detail = docs_detail([seen, new], sent_ids={doc_id(seen)})

    assert [hit["id"] for hit in detail["hits"]] == [doc_id(seen), doc_id(new)]
    assert list(detail["docs"]) == [doc_id(new)]
