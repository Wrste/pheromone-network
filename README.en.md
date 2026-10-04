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

## What can today's agents use it for?

In a typical agent workflow, the model handles understanding and generation while `pheromone_network` finds the historical information that is most useful right now. Place it before the context window as a local memory layer that changes with usage:

- **Long-term memory recall**: store user preferences, project conventions, past decisions, and durable facts, then inject only relevant memories into a new conversation.
- **Conversation context selection**: select a small set of high-scoring items from long chats, task traces, or observations to control context length and cost.
- **Tool result reuse**: remember search results, database queries, code analysis, or API calls and surface reusable results for similar tasks.
- **Task experience accumulation**: store the problem, actions taken, and outcome so similar tasks can recall a path that worked before.
- **User preference maintenance**: reinforce confirmed formats, wording, and working habits so responses better match an individual's workflow.
- **Reflection and failure memory**: keep failure causes, fixes, and validation results so later tasks can avoid a previously unsuccessful path.
- **Shared memory for multiple agents**: maintain separate kernels for a team, project, or workspace so agents can share facts and experience that proved useful in practice.

A simple integration loop is: **user input and task state -> query vector -> Top-K memory recall -> model context -> reinforce based on actual use**. It does not replace the model or generate answers; it helps the model reach relevant history faster.

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

## API reference

### `RecallKernel`

Use this API when you need online memory, candidate ranking, or a local recall layer.

```ts
new RecallKernel(dim: number, options?: RecallKernelOptions)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `codeDim` | `512` | Internal associative dimension, capped at `dim` |
| `maxNeighbors` | `8` | Maximum local connections per output |
| `tagCount` | `32` | Number of topology tags |
| `tagDistance` | `1` | Maximum tag distance for a connection |
| `config` | library defaults | Partial `LocalLearningConfig` override |
| `seed` | `1` | Deterministic initialization; use `null` for random initialization |

| Method or property | Returns | Purpose |
| --- | --- | --- |
| `observe(vector)` | `TrainStepReport` | Learn one memory vector by self-association |
| `score(query, candidate)` | `number` | Normalized similarity score; higher means closer |
| `encode(vector)` | `number[]` | Project a vector into the learned associative space |
| `evaporate(rate?)` | `number` | Decay pheromone by a rate in `[0, 1]`; returns affected connections |
| `decayByFactor(factor)` | `number` | Multiply pheromone by a factor below `1` |
| `pheromoneMass()` | `number` | Sum active long-term pheromone |
| `trained` | `boolean` | Whether at least one observation was made |
| `trainSteps` | `number` | Number of observations |
| `loss` | `number \| null` | Loss from the latest observation |
| `dim`, `codeDim`, `network` | property | Input dimension, internal dimension, and underlying network |

### Custom embedding in an agent

The kernel accepts any `number[]`, so an agent can use its existing embedding service and keep the original text beside the vector:

```ts
import { RecallKernel, type Embedder } from "pheromone_network";

const embed: Embedder = (text) => existingEmbeddingModel.embed(text);
const memory = new RecallKernel(1536, { codeDim: 256 });
const memories: Array<{ text: string; vector: number[] }> = [];

function remember(text: string) {
  const vector = embed(text);
  memory.observe(vector);
  memories.push({ text, vector });
}

function recall(query: string, limit = 3) {
  const queryVector = embed(query);
  return memories
    .map((item) => ({ text: item.text, score: memory.score(queryVector, item.vector) }))
    .sort((left, right) => right.score - left.score)
    .slice(0, limit);
}
```

Replace `existingEmbeddingModel.embed` with the embedding function already used by your application. The embedding dimension passed to `RecallKernel` must match the vectors you provide.

### `LocalPheromoneNetwork`

Use the lower-level engine when you need explicit input/output sizes, hidden layers, activations, tags, topology, or supervised local training.

```ts
import { LocalPheromoneNetwork, type Matrix } from "pheromone_network";

const network = new LocalPheromoneNetwork(4, [6], 2, {
  activation: "tanh",
  maxNeighbors: 3,
  tagCount: 2,
  tagDistance: 0,
  seed: 7,
});

const input: Matrix = [[1, 0, 0.5, -1]];
const target: Matrix = [[0.2, -0.4]];
const prediction = network.forward(input);
const report = network.localTrainStep(input, target);

console.log(prediction, report.loss, report.mode, report.activeSynapses);
```

`input` and `target` are row-major matrices. Their row counts must match; each input row has `inputSize` values and each target row has `outputSize` values. `forwardWithCache` additionally returns hidden activations and pre-activations. `localTrainStep` returns `TrainStepReport` with `loss`, `previousLoss`, `improved`, `mode`, `budgetPerOutput`, `activeSynapses`, and `prediction`. Pass `outputMask` to weight or disable output dimensions during training.

Network options include `maxNeighbors`, `tagCount`, `tagDistance`, `connectionRadius`, `activation`, `inputShape`, `hiddenShapes`, `outputShape`, `inputTags`, `outputTags`, `config`, and `seed`. Shapes support 1D or 2D layouts; `inputTags` and `outputTags` must match their layer sizes, and `hiddenShapes` must have the same length as `hiddenSizes`. `config` accepts a partial `LocalLearningConfig` and is merged with `DEFAULT_LOCAL_LEARNING_CONFIG`.

### `LocalPheromoneLayer`

Use a layer directly when building a custom topology or inspecting local connections. Its constructor is `new LocalPheromoneLayer(inFeatures, outFeatures, options?)`. The main methods are:

| Method | Purpose |
| --- | --- |
| `forward(matrix)` | Compute the layer output |
| `feedbackToInput(matrix)` | Propagate a post-signal to input features |
| `localUpdate(pre, post, localError, options)` | Apply a manual local Hebbian update |
| `connectedInputs(outputIndex)` | Inspect input indices connected to one output |
| `effectivePheromone()` | Read the short/long pheromone mixture |
| `pheromoneGate()` | Read the normalized pheromone gate |
| `effectiveWeight()` | Read weights after the pheromone gate |

Layer options support `maxNeighbors`, `connectionRadius`, `tagCount`, `tagDistance`, `inputShape`, `outputShape`, `inputTags`, `outputTags`, `usePheromoneGate`, `shortPheromoneWeight`, `longPheromoneWeight`, `bias`, and `seed`. The typed arrays exposed by a layer are live state; mutate them only when deliberately managing the engine.

Inspectable layer state includes `inFeatures`, `outFeatures`, `maxNeighbors`, `connectionRadius`, `tagCount`, `tagDistance`, `usePheromoneGate`, `neighborIndices`, `connectionMask`, `inputPositions`, `outputPositions`, `weight`, `bias`, `pheromone`, `shortPheromone`, `consolidation`, `lastUpdateMask`, `inputDegree`, and `lastDirectionalDerivative`.

### Vector utilities and shared types

| Export | Contract |
| --- | --- |
| `ngramEmbed(text, dim?)` | Deterministic character 2-5 gram vector, L2-normalized; default dimension `4096` |
| `cosine(left, right)` | Dot product for equal-length normalized vectors; returns `0` for different lengths |
| `normalizedCosine(left, right)` | True cosine similarity for vectors of any magnitude; compares the shared prefix when lengths differ |
| `Matrix` | `number[][]`, represented as rows of features |
| `Embedder` | `(text: string) => number[]` |
| `ActivationName` | `"tanh" \| "relu" \| "sigmoid" \| "identity"` |
| `DEFAULT_CODE_DIM` | `512` |
| `DEFAULT_RECALL_KERNEL_OPTIONS` | Default `RecallKernel` topology and seed |
| `DEFAULT_LOCAL_LEARNING_CONFIG` | Complete default training and pheromone configuration |

`LocalLearningConfig` controls learning rate, adaptive budget, evaporation, signal/weight clipping, short/long pheromone weights, and consolidation. Its fields are grouped as follows:

- **Learning and budget**: `learningRate`, `minBudgetPerOutput`, `maxBudgetPerOutput`, `initialBudgetPerOutput`, `shrinkFactor`, `growFactor`, `lossTolerance`.
- **Pheromone and stability**: `evaporation`, `pheromoneReinforcement`, `synapseDecay`, `signalClip`, `weightClip`, `neighborFollowDistance`, `minPheromone`, `maxPheromone`.
- **Timescales and consolidation**: `shortPheromoneEvaporation`, `longPheromoneEvaporation`, `shortPheromoneReinforcement`, `longPheromoneReinforcement`, `shortPheromoneWeight`, `longPheromoneWeight`, `consolidationStrength`, `consolidationGrowth`, `consolidationDecay`, `consolidationThreshold`, `consolidationLrFloor`, `consolidationLossGate`.

For the advanced `LocalPheromoneLayer.localUpdate` method, `LocalUpdateOptions` requires `budgetPerOutput`, `mode`, `config`, and `consolidate`.

The option interfaces are `RecallKernelOptions`, `LocalPheromoneNetworkOptions`, `LocalPheromoneLayerOptions`, and `LocalUpdateOptions`. The network also exposes `inputSize`, `hiddenSizes`, `outputSize`, `activation`, `maxNeighbors`, `config`, `layers`, `budgetPerOutput`, and `previousLoss` for inspection and monitoring.

## When is it not the right tool?

This is an association learning and candidate recall component. It is not a full text search engine, relational database, vector database, or general purpose classifier. Pair it with those systems when you need exact keyword filters, complex structured queries, large durable indexes, or strictly supervised prediction.

## Scenario test results

The reproducible cases in [`test/scenarios.test.ts`](test/scenarios.test.ts) use three candidates per query. Each candidate is observed once with `RecallKernel(256, { codeDim: 128, seed: 17 })` and the built-in `ngramEmbed`. The expected candidate ranked first in all four retrieval cases:

| Scenario | Expected top result | Top score | Next score | Result |
| --- | --- | ---: | ---: | --- |
| Tool cache | Matching `GET /api/orders/42` request | 0.9701 | 0.7412 | Pass |
| Agent memory | Concise English summary preference | 0.9106 | 0.4990 | Pass |
| Support tickets | Expired password-reset email link | 0.8025 | 0.5011 | Pass |
| Incident logs | Payment gateway timeout 504 | 0.8433 | 0.5746 | Pass |

The maintenance case also passed: `decayByFactor(0.5)` reduced pheromone mass from **984.6413** to **492.3206** after two observations. With the existing seven unit tests, `npm test` reports **12/12 passing**.

Run `npm test` to reproduce these cases and see the scores in the test output. These are small, deterministic examples of near-text matching and decay, not a benchmark or a measured recall rate on real data. The built-in character n-gram encoder does not establish semantic equivalence between different phrasings; use an appropriate embedding model when that is required. Recommendation, edge deployment, and game scenarios described above have not been measured by this suite.

## Learn more

- [中文完整文档](README.zh-CN.md)
- [Project home](README.md)

## License

This project is released under the [MIT License](LICENSE).
