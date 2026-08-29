# Fruity Fun — Filled-out RAG Framework

## 1. Problem and users

**User:** a child, parent, or teacher asking short or imaginative questions about fruit.

**Job:** provide a few readable, accurate lines grounded in the supplied PDF corpus and a cheerful original illustration.

**Non-goals:** medical advice, encyclopedic coverage, GraphRAG, knowledge-graph maintenance, or treating generated artwork as evidence.

## 2. Knowledge contract

The PDF corpus is authoritative for factual responses. Every factual answer must be supported by retrieved passages and use numbered citations. When retrieval confidence is below the configured threshold, the agent explicitly says the library lacks enough information and avoids invented facts. The visual path may imagine fruit characters, combinations, locations, and colors as long as it remains child-safe.

## 3. Ingestion

| Stage | Choice | Reason |
|---|---|---|
| Parsing | PyMuPDF page text | Fast, page-aware, dependable for ordinary PDFs |
| Cleanup | Collapse whitespace | Removes layout noise without rewriting facts |
| Chunking | ~150 words, ~25-word overlap | Small enough for precise fruit facts; overlap preserves boundaries |
| Identity | SHA-256 of source/page/position/content | Repeatable, idempotent upserts |
| Local store | JSON | Transparent corpus for BM25 and inspection |
| Dense store | Pinecone cosine index | Managed semantic retrieval at scale |
| Metadata | text, source, page, chunk | Citation and debugging support |

Scanned PDFs need OCR before ingestion; this is deliberately surfaced as a known limitation rather than silently guessing.

## 4. Retrieval

**Dense channel:** OpenAI query embedding against Pinecone, top 12. This handles paraphrases such as “outside dots” versus “external seeds.”

**Sparse channel:** BM25 over locally tokenized JSON chunks, top 12. This preserves exact fruit names, varieties, and rare terms.

**Fusion:** rank-based RRF with `k=60`. RRF avoids pretending incomparable Pinecone cosine and BM25 scores share a scale.

**Reranking:** an LLM orders the fused shortlist by direct relevance; deterministic RRF order remains the fallback.

**Final context:** top 8 passages.

## 5. Confidence

The score combines calibrated dense similarity (50%), normalized lexical strength (25%), cross-channel agreement (20%), and evidence coverage (5%). Cross-channel agreement is intentionally influential: a passage found semantically and lexically is safer to trust. `0.65` is the initial grounding threshold and should be calibrated against the evaluation set.

## 6. Generation and orchestration

LangGraph state nodes are:

1. `safety`: normalize/limit the request and redirect obviously unsafe themes.
2. `retrieve`: run dense and BM25 retrieval, tolerating dense-service failure.
3. `rerank`: rank fused candidates and set confidence/grounding state.
4. `answer`: generate short cited facts, or disclose insufficient evidence.
5. `picture`: create a safe original illustration; failure does not erase the answer.

This state makes failure behavior explicit and testable without GraphRAG complexity.

## 7. Answer contract

- Ages 6–12 reading style and 3–6 factual lines.
- No unsupported health claims or nutritional promises.
- Inline citations such as `[1]` map to source/page rows in the UI.
- Low-confidence answers disclose the gap rather than using model memory.
- Generated image content is not cited and is labeled as a picture.

## 8. Observability and evaluation

The UI exposes citation source, page, retrieval channels, fused score, confidence, and nonfatal technical warnings. Offline evaluation should log Recall@K, MRR/nDCG, citation correctness, groundedness, refusal appropriateness, age readability, picture safety, latency, and cost. No raw child prompt should be retained by default.

## 9. Security

Secrets stay in `.env`, which is ignored by Git. Retrieved PDF text is treated as evidence, not executable instruction. User prompts are bounded and the system message limits claims to supplied evidence. Production deployments should add rate limits, telemetry redaction, moderation, and an allowlist for corpus upload operators.
