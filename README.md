<div align="center">

# pheromone_network

**A zero-dependency pheromone memory kernel for agents.**

`TypeScript` · `Node >= 18` · `0 runtime deps`

[中文](README.zh-CN.md) | [English](README.en.md)

</div>

`pheromone_network` provides long-term memory through pheromone traces and sparse local connections. Frequently used links strengthen; idle links fade naturally.

## Quick start

```bash
npm install
npm run build
npm test
```

```ts
import { RecallKernel, ngramEmbed } from "pheromone_network";

const dim = 4096;
const kernel = new RecallKernel(dim, { codeDim: 512, seed: 1 });
const memories = ["prefers email", "invoice name is Acme", "billing cycle is net-30"];
const vectors = memories.map((text) => ngramEmbed(text, dim));

vectors.forEach((vector) => kernel.observe(vector));
const query = ngramEmbed("how long is the billing cycle", dim);
const ranked = memories
  .map((text, i) => ({ text, score: kernel.score(query, vectors[i]) }))
  .sort((a, b) => b.score - a.score);

console.log(ranked[0]);
```

## Documentation

- [中文文档](README.zh-CN.md)
- [English documentation](README.en.md)

## License

See the repository for license information.
