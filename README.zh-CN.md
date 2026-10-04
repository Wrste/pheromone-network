# pheromone_network

**一个会「用进废退」的零依赖信息素记忆内核，专为智能体设计。**

[中文](README.zh-CN.md) | [English](README.en.md) | [返回项目首页](README.md)

`TypeScript` · `Node >= 18` · `0 runtime deps`

## 一句话

`pheromone_network` 用「信息素 + 稀疏局部连接」为智能体提供长期记忆：被反复使用的连接会自动强化，长期闲置的连接会自然淡忘。

## 核心特点

| 特点 | 说明 |
| --- | --- |
| **零依赖** | 纯 TypeScript，仅使用 `node:crypto`，可以直接内联到其他项目 |
| **无需反向传播** | 使用手工 Hebbian 前向更新，轻量、可解释、易部署 |
| **稀疏局部连接** | 每个输出只连接少量 tag 兼容的输入，减少相互干扰 |
| **双时间尺度** | 同时维护短期痕迹、长期记忆和固化状态 |
| **用进废退** | 常被召回的连接增强，闲置连接按半衰期衰减 |
| **归一化召回** | 通过稀疏联想投影和归一化余弦相似度进行排序 |

## 30 秒上手

```ts
import { RecallKernel, ngramEmbed } from "pheromone_network";

const dim = 4096;
const kernel = new RecallKernel(dim, { codeDim: 512, seed: 1 });

// 写入：编码后进行自联想学习，重复使用的连接会被信息素强化
const memories = ["客户偏好邮件沟通", "发票抬头是 Acme 科技", "结算周期是月结 30 天"];
const vectors = memories.map((text) => ngramEmbed(text, dim));
vectors.forEach((vector) => kernel.observe(vector));

// 召回：将查询和候选记忆投影到同一联想空间后排序
const query = ngramEmbed("结算周期多久", dim);
const ranked = memories
  .map((text, i) => ({ text, score: kernel.score(query, vectors[i]) }))
  .sort((a, b) => b.score - a.score);

console.log(ranked[0]); // -> 结算周期是月结 30 天
```

## 接入任意智能体

1. **写入**：每沉淀一条长期记忆，调用 `kernel.observe(ngramEmbed(text, dim))`。
2. **召回**：每轮对话前，用 `kernel.score(query, candidate)` 对候选记忆排序，取 Top-K 注入上下文。
3. **维护**：定期调用 `kernel.evaporate(rate)`，或按时间调用 `kernel.decayByFactor(0.5 ** (deltaTime / halfLife))`。

可以选择以下任一种接入方式：

- **作为依赖**：使用 `npm link` 或复制项目目录，然后导入 `RecallKernel`。
- **直接内联**：将 `src/pheromone.ts`、`src/recall.ts` 和 `src/embed.ts` 复制到你的 `src/` 目录中。

## 项目结构

| 文件 | 职责 |
| --- | --- |
| `src/pheromone.ts` | 引擎核心：稀疏局部层、手工 Hebbian 更新和双尺度信息素 |
| `src/recall.ts` | 召回内核 `RecallKernel`：折叠、稀疏联想投影和归一化余弦打分 |
| `src/embed.ts` | 确定性字符 2-5 gram 编码器 `ngramEmbed` 与 `cosine` |
| `src/index.ts` | 统一导出 |

## 构建与测试

```bash
npm install
npm run build
npm test
```

## 实现来源

算法实现参考论文 **Local Pheromone Network: Sparse Local Learning with Multi-Scale Synaptic Trails, Consolidation, and Replay**（arXiv:2606.30669）。本 SDK 是该论文所述方法的零依赖 TypeScript 核心子集移植。

- 论文：<https://arxiv.org/abs/2606.30669>
- DOI：<https://doi.org/10.48550/arXiv.2606.30669>

## 主要 API

- 引擎：`LocalPheromoneNetwork`、`LocalPheromoneLayer`、`LocalLearningConfig`
- 召回：`RecallKernel`、`normalizedCosine`、`DEFAULT_CODE_DIM`
- 编码：`ngramEmbed`、`cosine`、`Embedder`
