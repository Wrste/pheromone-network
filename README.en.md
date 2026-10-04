# pheromone_network

**Memory that strengthens with use and fades with disuse.**

A zero-dependency, embeddable local association and recall kernel: frequently used relationships grow stronger, while idle relationships naturally decay.

[中文](README.zh-CN.md) | [English](README.en.md) | [Back to project home](README.md)

`TypeScript` · `Node >= 18` · `0 runtime deps` · `offline-friendly`

## What problem does it solve?

Many systems need to answer the same question: **given a new input, which pieces of past information are worth using again?**

`pheromone_network` packages that loop into a small local component. You provide vectors; it learns local associations, scores candidates, and lets unused relationships fade over time. It does not require a remote model, a vector database, a training dataset, a network connection, or a GPU.

Use it as a standalone recall layer, or place it in front of search, caching, recommendation, rules, or agent memory.

## Why use it?

- **Adapts through real usage**: repeated hits reinforce the relationships that matter in your product.
- **Lets stale context fade**: evaporation and time based decay keep old relationships from dominating forever.
- **Small and inspectable**: local connections, explicit weights, and cosine scores are easy to inspect.
- **Simple deployment**: zero runtime dependencies, no network requirement, and no GPU requirement.
- **Works with existing vectors**: use model embeddings, hashed features, or the built in `ngramEmbed` encoder.

## Where can it be used?

| Scenario | How to use it | Typical outcome |
| --- | --- | --- |
| **Semantic caching** | Associate request vectors with previous requests; reinforce reused entries and decay old ones | Fewer repeated computations for local APIs, tools, and generated content |
| **Recommendation and personalization** | Write behavior or content features; reinforce clicks, saves, and repeated actions | Lightweight interest ranking and personalized shortcuts |
| **Document, ticket, and knowledge deduplication** | Score a new item against historical vectors before review | Similar tickets, duplicate FAQs, and existing solutions surface early |
| **Log and event recall** | Encode event features or text n-grams and retrieve similar historical events | Faster incident investigation and runbook matching |
| **Offline and edge applications** | Keep associations on the device or inside a tenant or project scope | Desktop tools, browser extensions, IoT, and privacy sensitive systems |
| **Games and simulations** | Encode states, events, and behavior; reinforce paths that recur or work | NPC preferences, scene memory, strategy selection, and adaptive difficulty |
| **Pre filtering for retrieval** | Narrow a large candidate set locally before full text search, a vector database, or a remote model | Fewer downstream calls, shorter contexts, and smaller search ranges |

It is a strong fit when candidates already have vector representations, the system learns online, and stale relationships should lose influence automatically.

## How it works

```text
vector -> local association learning -> recall score -> usage feedback
                                               ^
                                      idle links decay over time
```

```text
Write:    observe(vector)
Recall:   score(queryVector, candidateVector)
Maintain: evaporate(rate) / decayByFactor(factor)
```

## 30-second start

```bash
npm install
npm run build
npm test
```

```ts
import { RecallKernel, ngramEmbed } from "pheromone_network";

const dim = 4096;
const memory = new RecallKernel(dim, { codeDim: 512, seed: 1 });
const records = [
  "prefers email",
  "invoice name is Acme",
  "billing cycle is net-30",
];
const vectors = records.map((text) => ngramEmbed(text, dim));

// Write new information. Reused relationships become stronger.
vectors.forEach((vector) => memory.observe(vector));

// Rank historical records for a new query.
const query = ngramEmbed("how long is the billing cycle", dim);
const ranked = records
  .map((text, index) => ({ text, score: memory.score(query, vectors[index]) }))
  .sort((left, right) => right.score - left.score);

console.log(ranked[0]);
// { text: "billing cycle is net-30", score: ... }

// Periodic maintenance keeps unused relationships from dominating.
memory.decayByFactor(0.98);
```

## When is it not the right tool?

This is an association learning and candidate recall component. It is not a full text search engine, relational database, vector database, or general purpose classifier. Pair it with those systems when you need exact keyword filters, complex structured queries, large durable indexes, or strictly supervised prediction.

## Learn more

- [中文完整文档](README.zh-CN.md)
- [Project home](README.md)

## License

See the repository for license information.
