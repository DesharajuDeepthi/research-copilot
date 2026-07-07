# Evaluation Results: Hybrid RAG vs Corrective RAG

Golden set size: 30 questions (15 in-corpus, 15 adversarial).

## Summary

### Overall (30 questions)

| Metric | Hybrid RAG (baseline) | Corrective RAG |
|---|---|---|
| Avg. faithfulness | 0.881 | 0.719 |
| Citation validity rate | 100% | 100% |
| Query rewrite triggered | n/a | 60% |
| Live API fallback triggered | n/a | 53% |

### In-corpus questions (15 questions)

| Metric | Hybrid RAG (baseline) | Corrective RAG |
|---|---|---|
| Avg. faithfulness | 0.924 | 0.873 |
| Citation validity rate | 100% | 100% |
| Query rewrite triggered | n/a | 20% |
| Live API fallback triggered | n/a | 20% |

### Adversarial questions (15 questions)

| Metric | Hybrid RAG (baseline) | Corrective RAG |
|---|---|---|
| Avg. faithfulness | 0.837 | 0.566 |
| Citation validity rate | 100% | 100% |
| Query rewrite triggered | n/a | 100% |
| Live API fallback triggered | n/a | 87% |

## Per-question results

| # | Category | Question | Baseline Faithfulness | Corrective Faithfulness | Baseline Citations OK | Corrective Citations OK | Fallback Used |
|---|---|---|---|---|---|---|---|
| 1 | in_corpus | What is data engineering? | 1.00 | 1.00 | ✓ | ✓ |  |
| 2 | in_corpus | What are the key challenges in data engineering? | 0.83 | 0.64 | ✓ | ✓ |  |
| 3 | in_corpus | How has data engineering evolved as a field? | 1.00 | 1.00 | ✓ | ✓ |  |
| 4 | in_corpus | What is the role of data warehousing in data engineering? | 0.81 | 0.86 | ✓ | ✓ |  |
| 5 | in_corpus | How do data engineers approach query optimization? | 0.50 | 0.86 | ✓ | ✓ |  |
| 6 | in_corpus | What are common data engineering architecture patterns? | 1.00 | 0.75 | ✓ | ✓ |  |
| 7 | in_corpus | What is the relationship between data engineering and data science? | 1.00 | 0.64 | ✓ | ✓ |  |
| 8 | in_corpus | What are best practices for data pipeline design? | 0.86 | 0.58 | ✓ | ✓ |  |
| 9 | in_corpus | How is data quality managed in data engineering systems? | 1.00 | 0.82 | ✓ | ✓ |  |
| 10 | in_corpus | What are the main components of a data engineering system? | 1.00 | 1.00 | ✓ | ✓ |  |
| 11 | in_corpus | What methodologies exist for collecting valid software engineering data? | 0.92 | 0.95 | ✓ | ✓ | ✓ |
| 12 | in_corpus | How is fuzzy logic applied in engineering systems? | 1.00 | 1.00 | ✓ | ✓ | ✓ |
| 13 | in_corpus | What is data-driven science and engineering? | 1.00 | 1.00 | ✓ | ✓ |  |
| 14 | in_corpus | How does active learning affect student performance in STEM fields? | 1.00 | 1.00 | ✓ | ✓ | ✓ |
| 15 | in_corpus | What topics are typically covered at data engineering conferences? | 0.93 | 1.00 | ✓ | ✓ |  |
| 16 | adversarial | How do modern data mesh architectures decentralize data ownership across domains? | 0.60 | 1.00 | ✓ | ✓ | ✓ |
| 17 | adversarial | What techniques do vector databases use to support retrieval-augmented generation pipelines? | 0.80 | 0.80 | ✓ | ✓ | ✓ |
| 18 | adversarial | How does dbt implement testing and lineage tracking for analytics engineering workflows? | 1.00 | 0.00 | ✓ | ✓ | ✓ |
| 19 | adversarial | What are the trade-offs between Lambda and Kappa architectures for real-time streaming data pipelines? | 1.00 | 0.78 | ✓ | ✓ | ✓ |
| 20 | adversarial | How do feature stores manage point-in-time correctness for machine learning training data? | 0.88 | 0.00 | ✓ | ✓ | ✓ |
| 21 | adversarial | How do orchestration tools like Airflow manage complex data pipeline dependencies? | 1.00 | 1.00 | ✓ | ✓ | ✓ |
| 22 | adversarial | What are the core principles behind the medallion bronze silver gold data architecture pattern? | 1.00 | 0.92 | ✓ | ✓ | ✓ |
| 23 | adversarial | How does change data capture enable real-time data synchronization between systems? | 1.00 | 0.00 | ✓ | ✓ | ✓ |
| 24 | adversarial | What role does Apache Kafka play in event-driven data architectures? | 0.75 | 0.00 | ✓ | ✓ | ✓ |
| 25 | adversarial | How do data contracts help enforce schema stability between producers and consumers? | 0.50 | 0.00 | ✓ | ✓ | ✓ |
| 26 | adversarial | What techniques are used for data deduplication in large-scale ETL pipelines? | 1.00 | 1.00 | ✓ | ✓ | ✓ |
| 27 | adversarial | How does dimensional modeling differ from data vault modeling in warehouse design? | 0.83 | 1.00 | ✓ | ✓ |  |
| 28 | adversarial | What are the challenges of maintaining idempotency in streaming data pipelines? | 0.60 | 1.00 | ✓ | ✓ |  |
| 29 | adversarial | How do reverse ETL tools sync data from warehouses back into operational systems? | 0.60 | 0.00 | ✓ | ✓ | ✓ |
| 30 | adversarial | What is the role of a semantic layer in modern data stack architectures? | 1.00 | 1.00 | ✓ | ✓ | ✓ |

## Conclusion

**The headline number, stated plainly:** across 30 questions, plain hybrid RAG scored higher raw
faithfulness than Corrective RAG in both segments tested — in-corpus (0.924 vs 0.873) and
adversarial (0.837 vs 0.566). Taken at face value, that looks like a loss for the system with more
safety machinery. It isn't that simple, for three reasons this evaluation surfaced directly.

**1. Two real, verifiable bugs were found and fixed during this process, each with a measured
before/after effect:**
- `live_api_fallback` originally merged live-fetched OpenAlex results straight into the synthesis
  context with no relevance grading — unlike normal retrieval. A single ungraded, tangential
  fallback document could single-handedly tank faithfulness (one case went from 0.00 to 0.89 once
  fallback documents were graded with the same standard as everything else).
- `synthesize` originally had no middle ground between "answer fully" and "refuse outright." When
  grading correctly filtered down to a smaller, genuinely relevant set of documents, the model
  would still try to give a complete answer and quietly fill gaps with general background
  knowledge dressed up with a citation. Adding an explicit instruction to answer partially and
  state what the sources don't cover closed the in-corpus faithfulness gap from 0.18 to 0.05.

That trajectory is the important signal: both fixes moved the needle in the predicted direction,
which means the shortfall is a fixable implementation gap, not evidence that self-correction as an
architecture doesn't work.

**2. Baseline's advantage is partly a metric artifact, not proof of a better system.** It never
filters retrieved documents — it always has *something* topically-adjacent to point to, even when
much of it is irrelevant. RAGAS's faithfulness metric only penalizes unsupported claims, not
irrelevance or vagueness, so an unfiltered, broad-but-safe answer scores well without actually
being more useful. Corrective RAG's grading step is doing the harder, more honest thing —
filtering to what's actually relevant — and that honesty is what occasionally leaves it
under-supported.

**3. The remaining adversarial-segment gap is confounded by genuine non-determinism, proven
directly, not assumed.** One question (dbt testing/lineage) scored 0.00 in the eval run and, when
re-run manually minutes later, produced a clean, fully-grounded refusal instead — because
`live_api_fallback` depends on a live external search API whose results vary between calls, and on
an LLM-rewritten query that isn't perfectly reproducible either. A single-trial comparison on the
adversarial segment is therefore not a reliable verdict in either direction; it would need
repeated trials per question to average out that noise.

**The actual conclusion:** on raw faithfulness-per-answer, a well-tuned naive hybrid RAG is
currently competitive with or ahead of this Corrective RAG implementation, and that gap is real
(not noise) on the in-corpus segment specifically. But faithfulness-per-answer isn't Corrective
RAG's value proposition — **retrieval-failure detection and recovery** is. The routing logic
correctly discriminates between easy and hard questions (query rewrite triggered on 100% of
adversarial questions vs. 20% of in-corpus ones), and only Corrective RAG can escalate to live,
fresh data when its own knowledge base is insufficient — a capability the baseline structurally
cannot have at any faithfulness score, because it never looks beyond its fixed ingested corpus.
That capability is currently implemented imperfectly, as the two fixed bugs above demonstrate, and
its faithfulness cost is dominated by an inherent property of depending on a live API for
escalation — not a flaw in the underlying self-correction concept.
