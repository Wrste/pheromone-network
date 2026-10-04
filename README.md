<div align="center">

# pheromone_network

**一个会「用进废退」的零依赖信息素记忆内核 —— 塞进任何智能体。**

*A zero-dependency pheromone memory kernel that lives by use and fades by disuse — drop it into any agent.*

`TypeScript` · `Node ≥ 18` · `0 runtime deps`

</div>

---

**中文** ｜ [**English**](#english)

---

# 中文

## 一句话

`pheromone_network` 用「信息素 + 稀疏局部连接」为智能体提供长期记忆：
**不训练、不反向传播**——被反复使用的连接自动强化，久不用的连接自然淡忘。

## 为什么值得一试

| 亮点 | 说明 |
| --- | --- |
| **零依赖** | 纯 TypeScript，仅用 `node:crypto`，可整包内联进任意工程 |
| **无需反向传播** | 手工 Hebbian 前向更新，轻量、可解释、随处可跑 |
| **稀疏局部** | 每个输出只连接一小撮 tag 相容输入，天生分区、抗干扰 |
| **双时间尺度** | 短期痕迹 + 长期记忆 + 固化（consolidation），模拟记忆巩固 |
| **用进废退** | 常被召回的连接变强，静置则按半衰期自然蒸发 |
| **归一化召回** | 编码 → 稀疏联想投影 → 归一化余弦打分，不被向量模长带偏 |

## 30 秒上手

```ts
import { RecallKernel, ngramEmbed } from "pheromone_network";

const dim = 4096;
const kernel = new RecallKernel(dim, { codeDim: 512, seed: 1 });

// 写入：编码后做自联想学习，被反复引用的连接会被信息素强化
const memories = ["客户偏好邮件沟通", "发票抬头是 Acme 科技", "结算周期是月结 30 天"];
const vectors = memories.map((text) => ngramEmbed(text, dim));
vectors.forEach((vector) => kernel.observe(vector));

// 召回：query 与候选投影到同一联想空间后打分排序
const query = ngramEmbed("结算周期多久", dim);
const ranked = memories
  .map((text, i) => ({ text, score: kernel.score(query, vectors[i]) }))
  .sort((a, b) => b.score - a.score);

console.log(ranked[0]); // → 结算周期是月结 30 天
```

## 接入任意智能体（三步）

1. **写入** —— 每沉淀一条长期记忆：`kernel.observe(ngramEmbed(text, dim))`
2. **召回** —— 每轮对话前：对候选记忆做 `kernel.score(query, candidate)` 排序，取 Top-K 注入上下文
3. **维护** —— 定期 `kernel.evaporate(rate)`，或按经过时间 `kernel.decayByFactor(0.5 ** (Δt / halfLife))`

两种引入方式，任选其一：

- **作为依赖**：`npm link` 或复制目录后 `import { RecallKernel } from "pheromone_network"`
- **整包内联**：把 `src/pheromone.ts`、`src/recall.ts`、`src/embed.ts` 三个文件拷进你的 `src/`，零配置即可编译

## 项目结构

| 文件 | 职责 |
| --- | --- |
| `src/pheromone.ts` | 引擎核心：稀疏局部层、手工 Hebbian、双尺度信息素（零 import） |
| `src/recall.ts` | 召回内核 `RecallKernel`：折叠 → 稀疏联想投影 → 归一化余弦打分 |
| `src/embed.ts` | 确定性字符 2–5 gram 编码 `ngramEmbed` 与 `cosine` |
| `src/index.ts` | 统一出口（`export *`） |

## 构建与测试

```bash
npm install
npm run build   # 编译到 dist/
npm test        # 7 项内核单测
```

## 它从哪来

算法实现参考论文 **Local Pheromone Network: Sparse Local Learning with Multi-Scale Synaptic Trails, Consolidation, and Replay**（arXiv:2606.30669）。
本 SDK 是该论文所述 **Local Pheromone Network** 的**零依赖 TypeScript 核心子集**移植：

| 论文机制 | 本项目实现 |
| --- | --- |
| 稀疏、按几何距离与 tag 兼容性限制邻居的局部层 | `LocalPheromoneLayer`（`maxNeighbors` / `tagCount` / `tagDistance`） |
| 手工 Hebbian 更新（不调用自动微分） | `LocalPheromoneLayer.localUpdate` · `LocalPheromoneNetwork.localTrainStep` |
| 多尺度突触痕迹（短程 + 长程信息素） | `shortPheromone` / `pheromone`（`shortPheromoneWeight` / `longPheromoneWeight`） |
| 固化（consolidation）状态 | `consolidation` 轨迹与 `updateConsolidation` |
| 在线自适应更新预算 | `LocalPheromoneNetwork.adaptBudget` |

> 论文中的结构可塑性、局部回放、GELU 等扩展，本核心子集有意未实现，聚焦「稀疏局部学习 + 多尺度信息素 + 固化」主干，并在其上叠加召回打分。

---

<a id="english"></a>

# English

## In one line

`pheromone_network` gives any agent long-term memory using **pheromone traces + sparse local connections**:
**no training, no backpropagation** — links that get used grow stronger, links left idle simply fade away.

## Why it's worth a look

| Highlight | What it means |
| --- | --- |
| **Zero dependencies** | Pure TypeScript, only `node:crypto`; inline it into any project |
| **No backpropagation** | A manual Hebbian forward update — light, explainable, runs anywhere |
| **Sparse & local** | Each output reads only a few tag-compatible inputs → natural partitioning |
| **Dual timescale** | Short-term trace + long-term memory + consolidation |
| **Use it or lose it** | Frequently recalled links grow; idle ones evaporate by half-life |
| **Normalized recall** | Encode → sparse associative projection → normalized cosine score |

## 30-second start

```ts
import { RecallKernel, ngramEmbed } from "pheromone_network";

const dim = 4096;
const kernel = new RecallKernel(dim, { codeDim: 512, seed: 1 });

// Write: encode, then learn by self-association (reused links get reinforced)
const memories = ["prefers email", "invoice name is Acme", "billing cycle is net-30"];
const vectors = memories.map((text) => ngramEmbed(text, dim));
vectors.forEach((vector) => kernel.observe(vector));

// Recall: project both sides into the same associative space, then score
const query = ngramEmbed("how long is the billing cycle", dim);
const ranked = memories
  .map((text, i) => ({ text, score: kernel.score(query, vectors[i]) }))
  .sort((a, b) => b.score - a.score);

console.log(ranked[0]); // → billing cycle is net-30
```

## Drop into any agent (3 steps)

1. **Write** — for each memory: `kernel.observe(ngramEmbed(text, dim))`
2. **Recall** — rank candidates with `kernel.score(query, candidate)` and inject the Top-K into context
3. **Maintain** — call `kernel.evaporate(rate)`, or decay over time with `kernel.decayByFactor(0.5 ** (dt / halfLife))`

Two ways to integrate:

- **As a dependency**: `npm link` or copy the folder, then `import { RecallKernel } from "pheromone_network"`
- **Inline it**: copy `src/pheromone.ts`, `src/recall.ts`, `src/embed.ts` into your `src/` — zero config, only needs `node:crypto`

## Project layout

| File | Responsibility |
| --- | --- |
| `src/pheromone.ts` | Engine core: sparse local layers, manual Hebbian, dual-timescale pheromone (zero imports) |
| `src/recall.ts` | Recall kernel `RecallKernel`: fold → sparse associative projection → normalized cosine |
| `src/embed.ts` | Deterministic char 2–5 gram encoder `ngramEmbed` + `cosine` |
| `src/index.ts` | Single public entry (`export *`) |

## Build & test

```bash
npm install
npm run build   # compile to dist/
npm test        # 7 kernel unit tests
```

## Where it comes from

Based on the paper **Local Pheromone Network: Sparse Local Learning with Multi-Scale Synaptic Trails, Consolidation, and Replay** (arXiv:2606.30669).
This SDK is a **zero-dependency TypeScript core subset** of that work.

- Paper: <https://arxiv.org/abs/2606.30669>
- DOI: <https://doi.org/10.48550/arXiv.2606.30669>

## Main API

- Engine: `LocalPheromoneNetwork`, `LocalPheromoneLayer`, `LocalLearningConfig`
- Recall: `RecallKernel`, `normalizedCosine`, `DEFAULT_CODE_DIM`
- Embedding: `ngramEmbed`, `cosine`, `Embedder`
