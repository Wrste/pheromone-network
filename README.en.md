# pheromone_network

**A zero-dependency pheromone memory kernel for agents.**

[中文](README.zh-CN.md) | [English](README.en.md) | [Back to project home](README.md)

`TypeScript` · `Node >= 18` · `0 runtime deps`

## In one line

`pheromone_network` gives agents long-term memory through pheromone traces and sparse local connections: frequently used links strengthen, while idle links fade naturally.

## Highlights

| Feature | Description |
| --- | --- |
| **Zero dependencies** | Pure TypeScript with only `node:crypto`; easy to embed in another project |
| **No backpropagation** | Manual Hebbian forward updates keep the kernel lightweight and explainable |
| **Sparse local connections** | Each output reads a small set of tag-compatible inputs, reducing interference |
| **Two timescales** | Short-term traces, long-term memory, and consolidation are maintained together |
| **Use it or lose it** | Frequently recalled links strengthen; idle links decay by half-life |
| **Normalized recall** | Candidates are ranked with sparse associative projection and normalized cosine similarity |

## 30-second start

```ts
import { RecallKernel, ngramEmbed } from "pheromone_network";

const dim = 4096;
const kernel = new RecallKernel(dim, { codeDim: 512, seed: 1 });

// Write: encode each memory, then reinforce it through self-association
const memories = ["prefers email", "invoice name is Acme", "billing cycle is net-30"];
const vectors = memories.map((text) => ngramEmbed(text, dim));
vectors.forEach((vector) => kernel.observe(vector));

// Recall: project the query and candidates into the same associative space
const query = ngramEmbed("how long is the billing cycle", dim);
const ranked = memories
  .map((text, i) => ({ text, score: kernel.score(query, vectors[i]) }))
  .sort((a, b) => b.score - a.score);

console.log(ranked[0]); // -> billing cycle is net-30
```

## Add it to an agent

1. **Write**: call `kernel.observe(ngramEmbed(text, dim))` for each durable memory.
2. **Recall**: before each conversation turn, rank candidates with `kernel.score(query, candidate)` and inject the Top-K into context.
3. **Maintain**: call `kernel.evaporate(rate)` regularly, or use `kernel.decayByFactor(0.5 ** (deltaTime / halfLife))` for time-based decay.

Choose either integration style:

- **As a dependency**: use `npm link` or copy the project directory, then import `RecallKernel`.
- **Inline**: copy `src/pheromone.ts`, `src/recall.ts`, and `src/embed.ts` into your `src/` directory.

## Project layout

| File | Responsibility |
| --- | --- |
| `src/pheromone.ts` | Engine core: sparse local layers, manual Hebbian updates, and dual-timescale pheromone |
| `src/recall.ts` | Recall kernel `RecallKernel`: folding, sparse associative projection, and normalized cosine scoring |
| `src/embed.ts` | Deterministic character 2-5 gram encoder `ngramEmbed` and `cosine` |
| `src/index.ts` | Single public entry point |

## Build and test

```bash
npm install
npm run build
npm test
```

## Origin

The implementation is based on **Local Pheromone Network: Sparse Local Learning with Multi-Scale Synaptic Trails, Consolidation, and Replay** (arXiv:2606.30669). This SDK is a zero-dependency TypeScript core subset of that work.

- Paper: <https://arxiv.org/abs/2606.30669>
- DOI: <https://doi.org/10.48550/arXiv.2606.30669>

## Main API

- Engine: `LocalPheromoneNetwork`, `LocalPheromoneLayer`, `LocalLearningConfig`
- Recall: `RecallKernel`, `normalizedCosine`, `DEFAULT_CODE_DIM`
- Embedding: `ngramEmbed`, `cosine`, `Embedder`
