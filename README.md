# AT-KV Assistant

Retrieval over the Austrian IT collective agreement (**IT-KV**) and related labour
law, answering with a citation to the exact paragraph.

**▶ [Interactive demo](https://timdev-r.github.io/atkv-demo.html)** — a recorded
replay of a real session, in the chat UI the service itself serves at `/`.

![The same question for 2026 and 2025 returns different figures with different citations, and an advice-seeking question is refused](docs/demo.svg)

```
Q: Wie hoch ist das Mindestgrundgehalt für ST1 Erfahrungsstufe?
A: ST1 / Erfahrungsstufe: 4.476 EUR brutto pro Monat  [IT-KV 2026, § 15]

Q: (same question, valid_year=2025)
A: ST1 / Erfahrungsstufe: 4.350 EUR brutto pro Monat  [IT-KV 2025, § 15]

Q: Soll ich meinen Arbeitgeber klagen?
A: (refused — the system documents the law, it does not advise on it)
```

Every figure in both demos comes from a live call to the service — see
[`docs/`](docs/). The terminal recording queries the running service directly,
so a retrieval regression breaks it; the chat replay is a verbatim transcript of
a real session, replayed rather than live because the deployed service scales to
zero and a first visitor would otherwise wait ~60 s for a cold start.

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
| dense only (e5-small, bilingual query) | 0.80 | 0.88 | 0.92 | 0.96 |
| lexical only (BM25 + decompounding) | 0.76 | 0.84 | 0.92 | 0.96 |
| **hybrid (four-way RRF)** | 0.80 | **0.92** | **0.92** | **0.96** |

Reranking with the cross-encoder no longer changes these numbers. The fast path
reaches what previously took ~1400 ms per query, so the flag remains off by
default.

**Every paragraph carries its own title, and that was the single biggest fix.**
RIS publishes an `<ueberschrift typ="para">` for each paragraph and the parser
was keeping only the `typ="g*"` Gliederung levels, silently discarding:

```
§ 3   Normalarbeitszeit
§ 4   Andere Verteilung der Normalarbeitszeit
§ 4a  Normalarbeitszeit bei Schichtarbeit
§ 9   Höchstgrenzen der Arbeitszeit
```

Without them those paragraphs are near-identical prose about *Normalarbeitszeit*
scoring within 0.007 of one another, and a question about the maximum daily
working time cannot find the paragraph literally titled *"Höchstgrenzen der
Arbeitszeit"*. Restoring them moved R@5 from 0.88 to 0.92, cross-lingual from
0.33 to 0.67, law from 0.50 to 0.67 and prose from 0.82 to 0.88 — for a
one-line parser change and no extra compute.

**Each query is issued in both languages, as four separate ranked lists**
(dense-EN, dense-DE, lexical-EN, lexical-DE). A bi-encoder prefers
same-language matches almost regardless of topical fit — the target of the
annual-leave question sat at dense rank **87** for the English query and rank
**1** for its German translation. Neither wins everywhere (a question the
English KV genuinely answers got *worse* in German, rank 2 → 10), so both are
searched and fused **by rank**. Merging them by score instead cost a question on
the language pairs: cosine scores from two different queries are not
comparable, which is why RRF is used here at all.

| | R@3 | R@5 | R@10 | language_pair |
|---|---|---|---|---|
| dense(EN) + lexical | 0.84 | 0.84 | 0.84 | 1.00 |
| max-merged dense + lexical | 0.84 | 0.84 | 0.88 | 0.83 |
| dense(EN) + dense(DE) + lexical | 0.88 | 0.88 | 0.92 | 1.00 |
| **four lists, one per language per retriever** | **0.92** | **0.92** | **0.96** | **1.00** |

The lexical query used to be the English and German forms concatenated, which
let BM25 match English chunks on the English half — two of three lists were
English-biased and outvoted the single list that could see a German-only
answer. Separate lists make the vote 2-2.

Three fusion rules (rank-sum, best-of-max, and both combined) then scored
**identically** — same totals, same categories, only a different pair of
questions failing. That is where fusion tuning stopped: at n=25 it would have
been rearranging noise.

Once the dense query is bilingual, **BM25 adds nothing measurable** — every
category is identical with and without it, and the R@1/R@10 differences are one
question each, on *different* questions. It is kept because it is a genuinely
different signal and because the compound-splitting measurement depends on it,
not because the numbers justify it.

**Hybrid does not beat lexical alone here**, and that is an honest negative
result rather than a tuning opportunity. RRF uses rank only; when one retriever
is right and the other confidently wrong, it splits the difference. At n=25 the
R@10 gap is 2 questions, so the defensible statement is "fusion showed no
benefit on this set", not "fusion is bad".

Reranking with a multilingual cross-encoder reaches **R@5 0.88 / R@10 0.96** at
**~1400 ms** against ~4 ms — a trade-off, not a free win, so it sits behind a
flag (`rerank: true`).

### 4. Answers and citations

Two metrics, because a number can be right by coincidence:

- **loose** — the expected figure or phrase appears in the answer.
- **attributed** — it appears **and** the expected paragraph was in the context.

`attributed` is the one to trust. Asked for the maximum daily working time
(AZG § 9, twelve hours), the model once answered *"…is 12 hours [IT-KV 2026,
§ 4]"* — right number, wrong law, with the correct paragraph never retrieved.
For a system whose whole claim is provenance, an answer that cannot be
attributed is luck rather than correctness.

| provider | k | loose | attributed |
|---|---|---|---|
| **ollama `qwen2.5:3b-instruct`** | **3** | **19/25 (0.76)** | **18/25 (0.72)** |
| ollama `qwen2.5:3b-instruct` | 5 | 17/25 (0.68) | 17/25 (0.68) |
| extractive (top chunk verbatim, no LLM) | 3 | 19/25 (0.76) | 18/25 (0.72) |
| extractive | 5 | 19/25 (0.76) | 18/25 (0.72) |

**Context size matters more than the model.** At k=5 the generator scored two
questions below the extractive baseline; at k=3 it matches it exactly, at no
cost in recall — the expected paragraph dropped out of the context for zero
questions. Both failures it fixed were *selection* errors among correct
candidates:

```
cov_urlg_36_werktage   k=5  "30 Werktage [UrlG, § 2]."   <- UrlG § 2 states BOTH 30 and 36
                       k=3  "36 Werktage [UrlG, § 2]."
```

Three failures survive at any k, and they are model-quality problems rather
than retrieval or context ones: two questions refused with § 13 sitting in the
context, and one answer that quoted § 3 while dropping *"dreimonatigen
Kündigungsfrist"* — the entire substance of the answer.

**What the generator is actually for.** Extractive can only return rank 1, so
it fails whenever the best chunk is not the right one: asked for the LT salary
it returned the § 15 prose chunk, while the model found the cell at rank 2.
That is the trade — the model's freedom to choose among candidates fixes
rank-1 misses and creates selection errors. Narrowing the context to k=3 keeps
the first and largely removes the second.

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
make docker     # build the image (2.6 GB; fetches sources, bakes index + models)
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

### Container

`make docker` produces a **2.6 GB** image that starts in **11.7 s** and needs no
network at boot: the FAISS index and every model the service loads are baked in
at build time. With `minReplicas: 0` in Stage 2, a cold start that downloaded
model weights would charge that cost to a user rather than to a deploy.

Two things that cost real time to find:

- **`UV_EXTRA_INDEX_URL` does not give you CPU-only torch.** `uv sync --frozen`
  honours the lockfile, and PyPI's torch declares its CUDA dependencies with the
  marker `platform_system == "Linux"` — no architecture gate. This **ARM** image
  therefore pulled 3.3 GB of x86-only nvidia wheels: 7.81 GB total, 42% of it
  unusable. Pinning torch to `download.pytorch.org/whl/cpu` for Linux in
  `[tool.uv.sources]` (which does publish `manylinux_2_28_aarch64` wheels) took
  it to 2.6 GB.
- **Bake every model, not just the obvious one.** Caching only the embedder left
  the container downloading the translation model at boot — 49.7 s to ready and a
  hard dependency on reaching huggingface.co. Now 11.7 s and fully offline.

Ingest retries transient HTTP failures with backoff. A dropped connection from
wko.at killed a 34-minute build once; 4xx other than 429 still fail immediately,
since a 404 will not fix itself.

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

- **Sibling paragraphs are the remaining failure, in both languages.** The two
  questions still missed at R@5 both want AZG § 3 and get §§ 4 / 4a — one asked
  in German, one in English, so this is no longer a cross-lingual problem. Those
  paragraphs all concern *Normalarbeitszeit* and score within 0.007 of each
  other. BM25 makes it worse rather than better: the correct answer states the
  rule once and tersely while the derogations elaborate, so § 3 has the **lowest**
  term frequency of any candidate. Term-frequency ranking is systematically
  biased against defining provisions.
- **Cross-lingual is largely addressed, at two price points.** Bilingual dense
  querying takes it from 0.00 to 0.33 for ~20 ms; the cross-encoder takes it to
  0.67 for ~1400 ms. The two are redundant — with reranking on, the bilingual
  query adds nothing, because both fix the same bi-encoder language bias. The
  cheap one is on by default.
- **Source stratification was tried and failed.** The hypothesis — that KV
  chunks crowded law chunks out of the pool — was wrong: the pool already
  contained the target for 24 of 25 questions. The problem is ranking within
  the pool, not admission to it.
- **A 3B generator adds little over the retrieved text.** It matches the
  extractive baseline exactly at k=3 rather than beating it, because step 4
  verbalises each table cell into a sentence that is already an answer. It
  earns its place on rank-1 misses and on prose that needs rephrasing, not on
  table lookups.
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
