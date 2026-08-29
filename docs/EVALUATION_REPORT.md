# Submission and Evaluation Report

## Scope delivered

The submission implements PDF extraction, JSON persistence, Pinecone dense ingestion, local BM25, hybrid RRF, LLM reranking, confidence gating, LangGraph state flow, kid-friendly answer generation, safe image generation, Streamlit chat, configuration, automated unit tests, and setup documentation. GraphRAG and Neo4j are excluded by design.

## Evaluation protocol

Build a labeled set of at least 60 prompts:

- 20 exact-name or exact-fact questions (tests BM25).
- 15 paraphrased questions (tests dense retrieval).
- 10 multi-fruit questions.
- 10 questions absent from the corpus (tests confidence behavior).
- 5 unsafe or adversarial picture requests.

For each prompt, label relevant source pages and whether a factual answer is allowed. Score retrieval with Recall@8 and nDCG@8; generation with citation precision, claim support, age-appropriate readability, and correct low-confidence disclosure. Have an adult reviewer inspect every image for child safety and prompt alignment.

## Acceptance targets

| Measure | Target |
|---|---:|
| Retrieval Recall@8 | ≥ 0.90 |
| Citation precision | ≥ 0.95 |
| Supported factual claims | ≥ 0.95 |
| Correct insufficient-evidence behavior | ≥ 0.90 |
| Unsafe-image prevention | 100% on red-team set |
| Median answer latency (excluding image) | ≤ 5 s |

Targets require a real corpus and credentials; they are not fabricated as completed measurements in this repository.

## Failure analysis

| Failure | Likely cause | Current behavior | Improvement |
|---|---|---|---|
| No passages found | Missing JSON, absent topic, poor PDF text | Honest no-evidence answer; image can continue | Verify ingestion; add OCR; expand corpus |
| Exact fruit variety missed | Tokenization or spelling variation | Dense search may recover it | Add synonyms or character n-grams |
| Plausible but wrong passage ranks first | Ambiguous query | RRF + reranker reduce risk | Add a cross-encoder and query rewriting |
| Confidence too optimistic | Small/easy calibration set | Cross-channel agreement is required for high score | Calibrate threshold with reliability curves |
| Scanned PDF yields no content | No OCR layer | Ingestion produces no useful page chunks | Add OCR and page-quality checks |
| Pinecone outage | Network or service failure | Local BM25 still works; warning recorded | Retry/circuit breaker and cached dense results |
| Reranker output malformed | Model returns non-JSON | Keeps deterministic RRF order | Use structured output schema |
| Image request rejected or times out | Provider policy/service issue | Text answer remains visible | Retry once with simpler safe prompt |
| Prompt injection inside PDF | Corpus contains instructions | System limits evidence to facts, but risk remains | Ingestion-time classification and quoted context delimiters |
| Unsupported health advice | Question invites medical claim | System forbids medical promises | Add dedicated medical-intent policy node |

## Test coverage supplied

Unit tests cover overlapping chunk windows, empty input, BM25 keyword behavior, fusion ordering, cross-channel agreement, and confidence. Integration tests needing paid services are intentionally separate from local unit tests and should run against a disposable namespace.

## Recommended next evaluation run

After credentials and PDFs are available: ingest the corpus, manually verify random JSON chunks against their source pages, run the 60-prompt labeled set, chart confidence versus claim support, adjust `CONFIDENCE_THRESHOLD`, then conduct picture-safety review. Record corpus hash, model versions, configuration, date, latency, token usage, and image cost so results are reproducible.
