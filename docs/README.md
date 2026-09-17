# Demo assets

`demo.svg` — animated terminal recording, embeddable in a README or on a
GitHub Pages site. No JavaScript, no hosting, no dependency on the deployed
service being awake.

## Regenerating it

Every figure in the recording comes from a live call to the running service.
Nothing is printed from a fixture, so if retrieval regresses the recording
breaks — which is the only way a demo stays honest as the code changes.

```bash
make serve                       # in one shell
./docs/demo.sh                   # check the real output first
TERM=xterm-256color asciinema rec --overwrite --cols 100 --rows 26 \
  --command ./docs/demo.sh docs/demo.cast
npx --yes svg-term-cli --in docs/demo.cast --out docs/demo.svg \
  --window --width 100 --height 26 --padding 14
```

Requires a running Ollama with `qwen2.5:3b-instruct`; without it the service
falls back to the extractive provider and the answers read differently (the
figures and citations stay correct).

`demo.cast` is committed so the SVG can be re-rendered — different size,
padding, theme — without re-running the service.


## `demo-chat.html` — the chat replay

Self-contained: no CDN, no build step, no framework. Works from `file://`, from
GitHub Pages, or in an `<iframe>`.

It is a **replay**, not a live client, and the page says so. The transcript in
the `TRANSCRIPT` constant was captured verbatim from a real session — answers,
citations, page numbers and latencies included. The reason it replays rather
than calls the API is `minReplicas: 0`: the deployed service has nothing running
between requests, so the first visitor after idle waits for a 2.6 GB image pull.
On a portfolio link that reads as broken, not as cost-efficient.

The live version of the same UI is served by the app itself at `/`
(`src/atkv/static/index.html`), where the year selector makes the version filter
something you do rather than something you read about.

### Putting it on a GitHub Pages site

```html
<iframe src="atkv-demo.html" style="width:100%;height:640px;border:0"
        title="AT-KV Assistant demo"></iframe>
```

Or copy `demo-chat.html` to the Pages repo and link it directly — it has no
dependencies.
