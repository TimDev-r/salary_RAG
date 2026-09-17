# AT-KV Assistant

Retrieval over the Austrian IT collective agreement (**IT-KV**) and related labour
law, answering with a citation to the exact paragraph.

```
Q: Wie hoch ist das Mindestgrundgehalt für ST1 Erfahrungsstufe?
A: ST1 / Erfahrungsstufe: 4.476 EUR brutto pro Monat  [IT-KV 2026, § 15]

Q: (same question, valid_year=2025)
A: ST1 / Erfahrungsstufe: 4.350 EUR brutto pro Monat  [IT-KV 2025, § 15]

Q: Soll ich meinen Arbeitgeber klagen?
A: (refused — the system documents the law, it does not advise on it)
```

Everything runs locally. Total cost: **€0**. No paid APIs, no cloud resources,
no trials.

---

## Why this is not another chat-with-your-PDF

The 2025 and 2026 agreements are ~95% identical text on the same page numbers.
Only the numbers move. An embedding model cannot tell them apart, and will
return the wrong year **confidently**. That is the problem this project is
built around, and it is measured rather than asserted.

---

## Measured results

All figures from `make eval` on a frozen set of 25 answerable + 5 refusal
questions, every one traced to a source document.

> **n = 25.** One question is **4 percentage points**. A one- or two-question
> difference is not distinguishable from noise, and nothing below should be
> read as if it were.

### 1. Version filter — the headline result

A hard metadata filter on `valid_from`, applied **before ranking**.

| | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| filter OFF | 0.64 | 0.84 | 0.84 | 0.84 |
| **filter ON** | **0.80** | 0.84 | 0.84 | 0.84 |
| difference | **+0.16** | +0.00 | +0.00 | +0.00 |

The gain is at **rank 1 and nowhere else**. The right year was already in the
top few; the wrong year was sitting on top of it.

**Pre- vs post-filtering is not a detail.** On an adversarial case — 8
near-duplicate 2025 chunks scoring above 4 chunks from 2026 — post-filtering
returns **0 of 5** requested results while pre-filtering returns all 4 that
exist. Post-filtering does not rank worse; it silently returns nothing.

### 2. German-aware lexical search

BM25 alone, with and without corpus-driven compound splitting:

| | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| stemming only | 0.76 | 0.80 | 0.84 | 0.88 |
| **+ decompounding** | 0.76 | **0.88** | **0.88** | **0.92** |
| difference | +0.00 | **+0.08** | +0.04 | +0.04 |

Snowball stemming normalises inflection (`Gehälter → Gehalt`) but leaves
compounds whole. A user asks about `Gehalt`; the document says
`Mindestgrundgehalt`; plain BM25 sees no shared token at all and scores **zero**.

```
mindestgrundgehalt  ->  ['gehalt']
normalarbeitszeit   ->  ['normal', 'arbeitszeit']
urlaubsausmaß       ->  ['urlaub', 'ausmaß']
```

Splits use only parts that occur as standalone words elsewhere in this corpus —
no dictionary is downloaded, and the split is domain-adapted. German compounds
are right-headed, so when full decomposition fails the head noun is still
extracted. The head must be a **noun**, detected for free from capitalisation
(German capitalises nouns; verbs only sentence-initially) — without that check,
`lehrlingseinkommen` split to `kommen`, the verb.

### 3. Retrievers

| | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| dense only (e5-small) | **0.80** | 0.84 | 0.84 | 0.84 |
| lexical only (BM25 + decompounding + query translation) | 0.76 | **0.88** | **0.88** | **0.92** |
| hybrid (RRF) | 0.80 | 0.84 | 0.84 | 0.84 |

**Hybrid does not beat lexical alone here**, and that is an honest negative
result rather than a tuning opportunity. RRF uses rank only; when one retriever
is right and the other confidently wrong, it splits the difference. At n=25 the
R@10 gap is 2 questions, so the defensible statement is "fusion showed no
benefit on this set", not "fusion is bad".

Reranking with a multilingual cross-encoder reaches **R@5 0.88 / R@10 0.96** at
**~1400 ms** against ~4 ms — a trade-off, not a free win, so it sits behind a
flag (`rerank: true`).

### 4. Answers and citations — the language model is a net negative here

| | answer accuracy | citation accuracy | per question |
|---|---|---|---|
| **extractive** (top chunk verbatim, no LLM) | **19/25 (0.76)** | 21/25 (0.84) | ~0 s |
| ollama `qwen2.5:3b-instruct` | 17/25 (0.68) | 21/25 (0.84) | 26.6 s |

A 3B model scores **two questions worse** than returning the retrieved passage
unchanged, and takes 26 seconds to do it. That is an uncomfortable result and
it is reported because it is what was measured; at n=25 the gap is not
statistically strong, but the failures are specific and diagnosable rather
than random.

Splitting the 8 wrong answers by where they went wrong:

| | | |
|---|---|---|
| **retrieval** | 4 | the cross-lingual and sibling-paragraph cases already listed under limitations |
| **generation** | 4 | `lang_sonderzahlung_de/en` refused although § 13 was in context; `cov_kuendigungsfrist_kv` quoted § 3 but dropped *"dreimonatigen Kündigungsfrist"*; `cov_urlg_36_werktage` read **30** out of UrlG § 2 when the question asked for the figure after 25 years of service — the paragraph contains both numbers |

That last one is not bad luck. The eval set deliberately pairs two questions
against a paragraph holding both 30 and 36 Werktage, precisely because
returning it is a retrieval success and an answering failure. It caught
exactly what it was built to catch.

**Why extractive wins.** Step 4 verbalises every table cell into a sentence, so
the retrieved chunk *is already an answer*: `ST1 / Erfahrungsstufe: 4.476 EUR
brutto pro Monat`. There is nothing for a model to add, and three ways for it
to subtract — refuse, omit, or pick the wrong number from a paragraph that
holds several. The generator earns its place on prose questions that need
rephrasing; on a table lookup it is a liability.

The service therefore ships with both, routes to the model when one is
reachable, and falls back to extractive when it is not — and the fallback is
not a degraded mode so much as the higher-scoring one on this corpus.

### Refusals

| | |
|---|---|
| refusals caught | **5/5** |
| false refusals on factual questions | **0/25** |

Refusal is rule-based, so a decision is deterministic and reproducible. The
discriminator is grammatical — *"soll ich"* / *"should I"* — not topical:
whether the user asks what the documents **say** or what they should **do**.
Refusing on legal vocabulary fails both ways: *"Mit welcher Frist kann der
Kollektivvertrag gekündigt werden?"* is pure fact, and *"Habe ich Anspruch auf
ein 13. Monatsgehalt?"* is the system's core use case.

### 5. Model size — a useful negative result

| | R@1 | R@5 | R@10 | index build | query |
|---|---|---|---|---|---|
| **e5-small** (118M, 384d) | 0.80 | 0.84 | 0.84 | **13 s** | **29 ms** |
| e5-large (560M, 1024d) | 0.80 | 0.88 | 0.92 | 506 s | 122 ms |

Identical at R@1. One to two extra questions at R@5/R@10 for **38× the build
cost and 4× the query latency** — not separable from noise at this sample size.
`e5-small` is the default on that evidence.

---

## Architecture

```
question
   │
   ├─ guard ─────────→ refuse advice-seeking questions (before any index is touched)
   ├─ filter ────────→ valid_from/valid_to + tenant_id, applied PRE-ranking
   ├─ retrieve ──────→ dense (FAISS) ∥ lexical (BM25, German-aware)
   ├─ fuse ──────────→ reciprocal rank fusion → candidate pool (50)
   ├─ rerank ────────→ optional cross-encoder
   ├─ cap ───────────→ ≤3 per section, top k
   └─ generate ──────→ Ollama, or extractive fallback
```

**Corpus** — 538 chunks: 4 collective agreements (DE/EN × 2025/2026, 344 chunks)
and AZG + UrlG via the RIS OGD API (194 chunks).

**Table handling.** Most questions resolve to one cell of the salary grid in
§ 15. `extract_text()` flattens it to `Einstiegsstufe 2 236 2 547 3 267 4061
5 301` — the thousands separator is a space, so five values become nine tokens,
and the spacing is not even internally consistent. Worse, the `Berufseinsteiger`
row is ragged, so reading numbers in text order shifts every value one column
left. `pdfplumber`'s lattice detection recovers the true alignment from the
PDF's 120 cell rectangles. Each cell is then **verbalised** into a sentence,
which turns a geometry problem into a text problem:

```
IT-KV 2026, § 15 Tätigkeitsfamilien, Vorrückungsstufen und Mindestgrundgehälter
Die Mindestgrundgehälter betragen ab 1.1.2026:
ST1 / Erfahrungsstufe: 4.476 EUR brutto pro Monat
```

**Provenance.** The WKO PDFs are the social partners' copyrighted documents:
they are fetched at ingest time from `manifests/sources.yaml` and **never
committed**. Every download is checked for an expected string on page 1 — the
project's most important guard, because a wrong-year agreement passes every
other check and is indistinguishable from a correct one downstream.

---

## Running it

```bash
make ingest     # fetch sources, build the index (cached afterwards)
make serve      # API on :8000, OpenAPI docs at /docs
make test       # ground-truth, guard and generation tests
make eval       # the numbers above
make eval-fast  # same, skipping generation (no model server needed)
```

Generation is optional. With no model server reachable the service routes to an
**extractive** provider that returns the top source passage verbatim with its
citation — it cannot hallucinate, because it cannot write.

### API

| | |
|---|---|
| `POST /query` | `{question, valid_year?, lang, k, rerank}` → answer + citations + `trace_id` |
| `POST /query/stream` | server-sent events; **citations arrive before the answer** |
| `POST /ingest` | rebuild the index from the manifest |
| `GET /healthz` | status, corpus coverage, stale source types |

`valid_year` is optional — omitted, it means **whatever is in force today**.
If the requested date falls outside the agreement's coverage, the response
carries an explicit `notice` naming the window. Nothing out-of-window is ever
substituted: serving the 2026 table in 2027 would be a wrong number with a real
citation attached.

### Replayability

Every query logs a `trace_id` with the retrieved chunk ids and their component
scores kept **separate**, so "which retriever dragged that in" is answerable
from the log without re-running anything.

```json
{"msg":"query","trace_id":"ed2df4d6ab7b","as_of":"2026-07-01","latency_ms":720.1,
 "retrieved":[{"doc_id":"it-kv-2026-de","section_ref":"§ 15",
               "dense":0.9048,"lexical":15.13,"fused":0.0328}]}
```

---

## Known limitations

- **Cross-lingual retrieval is the weakest area.** An English question about
  UrlG § 2 still retrieves the English KV. Query translation raised the pool
  ceiling from 0.96 to 1.00 and cross-lingual R@5 from 0.00 to 0.67; the
  remainder is unsolved.
- **Sibling paragraphs are not separable.** AZG §§ 3, 4, 4a, 4b all concern
  *Normalarbeitszeit* and score within 0.007 of each other. BM25 makes it
  worse, not better: the correct answer (§ 3) has the **lowest** term frequency,
  because a defining provision states the rule once while the derogations
  elaborate at length. Term-frequency ranking is systematically biased against
  definitions.
- **Source stratification was tried and failed.** The hypothesis — that KV
  chunks crowded law chunks out of the pool — was wrong: the pool already
  contained the target for 24 of 25 questions. The problem is ranking within
  the pool, not admission to it.
- **Generation is slow, and the latency figures are host-dependent.** The
  numbers above were measured at ~10.8 tok/s with 87 tok/s prompt evaluation,
  on a machine carrying ~15 GB of swap against 8 GB of RAM. After closing a
  browser that had accumulated ~8.6 GB of footprint over 34 days of uptime and
  rebooting, prompt evaluation rose to **136 tok/s — 57% faster with no code
  change**. Wall-clock timings for the same query varied threefold between runs
  under pressure, which is why model and context decisions in this project were
  made on token counts rather than seconds. Streaming puts the first token on
  screen in ~3 s instead of ~11 s, but does not make anything faster.
- **Law versioning is single-version.** RIS resolves statute versions
  server-side at fetch time, so the index holds one consolidated version;
  `doc_id` does not encode the Fassung date.

---

## Stack

Python 3.12 · FastAPI · pdfplumber · sentence-transformers (`multilingual-e5-small`)
· FAISS · rank-bm25 + snowball · `mmarco-mMiniLMv2` cross-encoder ·
`opus-mt-en-de` · Ollama (`qwen2.5:3b-instruct`) · pytest · Docker

Sources: [RIS OGD API](https://data.bka.gv.at/ris/api/v2.6) (open government
data) and [WKO](https://www.wko.at/oe/kollektivvertraege/it-dienstleistungen-informationstechnologie-datenverarbeitu)
(fetched at ingest, never committed).
