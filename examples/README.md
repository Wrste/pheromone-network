# Browser demo

`index.html` is a static browser demo for the published `pheromone_network` package. It loads version `0.1.0` from `esm.sh`, so no local build step or server dependency is required for the demo itself. The demo uses the package's `RecallKernel` and a small synchronous browser encoder because the package's built-in `ngramEmbed` currently relies on Node's `crypto` API. The displayed recall score combines kernel similarity with a capped logarithmic reuse-frequency bonus, so repeated observations influence ranking without overwhelming content similarity. The decay control uses a half-life model: `factor = 0.5 ** (elapsed hours / half-life hours)`, then calls the package's existing `decayByFactor` method.

## Run locally

From the project root:

```bash
npx serve examples
```

Then open the URL printed by `serve`. Opening the HTML directly with `file://` may be blocked by browser module security rules.

## Publish with GitHub Pages

The repository includes [`.github/workflows/pages.yml`](../.github/workflows/pages.yml), which publishes `examples/` as a GitHub Pages artifact whenever `main` changes. In the repository settings, open **Pages**, set the source to **GitHub Actions**, and run the workflow once. The demo will then be available at:

```text
https://wrste.github.io/pheromone-network/
```

The page keeps all entered memories in browser memory only. It does not upload them to a server.

## Encoder limitations

The built-in browser encoder is a deterministic character N-gram hash. It is useful for short text, near-duplicate matching, logs, and offline prototypes, but it does not understand meaning. It can over-rank text that shares common phrases and miss paraphrases with different wording.
