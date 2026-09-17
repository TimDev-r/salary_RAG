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
