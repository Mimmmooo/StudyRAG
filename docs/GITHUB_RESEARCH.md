# GitHub RAG reference review (November 2026)

StudyRAG is a local-first notes Q&A app, not an autonomous agent platform. We reviewed:

| Project | Useful idea | Decision for StudyRAG |
| --- | --- | --- |
| [500 AI Agents Projects](https://github.com/ashishpatel26/500-AI-Agents-Projects) (MIT) | Education agents and document Q&A examples, with a directory of external implementation links | Reference only. Adding a multi-agent orchestration framework would not solve current retrieval accuracy. |
| [Kotaemon](https://github.com/Cinnamon/kotaemon) (Apache-2.0) | Hybrid full-text and vector retrieval to improve recall | Apply the hybrid-search pattern within existing ChromaDB pipeline. |
| [RAGFlow](https://github.com/infiniflow/ragflow) (Apache-2.0) | Document-centric retrieval and grounding | Keep page-aware text evidence and indexing workflow; avoid copying its infrastructure. |
| [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques) | Retrieval fusion, reranking, and retrieval evaluation patterns | Use as design guidance and test with Bengali/English questions. |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0) | OCR and structured PDF extraction | Potential future optional OCR engine; not included as a dependency without Bengali document benchmark evidence. |
| [Haystack](https://github.com/deepset-ai/haystack) (Apache-2.0) | Modular retrieval and generation pipelines | Existing modules already separate the necessary responsibilities. |

## Practical improvement

StudyRAG already combines vector and Bengali keyword search, but keyword retrieval previously loaded every document from ChromaDB **on every question** and matched arbitrary substrings. This is especially wasteful when a long scanned book contains many chunks.

This change builds a **cached, inverted keyword index** the first time a session searches, shared by later queries in that session. It is invalidated whenever chunks are inserted, removed, or cleared. Search matches **complete Bengali/English words**, ranks hits by corpus rarity and query coverage, and favors contiguous multiword phrases. The existing vector retrieval remains available when OCR changes spelling or lexical terms do not match.

This improves the retrieval implementation without requiring a larger model, agent framework, external API, new dependency or changing the citation-free chat UI. It does **not** guarantee correct answers when OCR or generation is wrong. Offline tests cover story titles, English questions, false partial-word matches, cache reuse, and document replacement.

No third-party source code or model weights were copied. The linked repositories are acknowledged for the design ideas; StudyRAG's implementation and tests are original.

## Local validation

```bash
python -m ruff check .
python -m pytest -q
```

For end-to-end accuracy, compare the answer with the original PDF for both Bengali title/author and longer explanation questions. Synthetic unit tests cannot establish OCR recognition accuracy.
