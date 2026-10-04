# pheromone_network

**让记忆会增强，也会淡忘。**

一个零依赖、可嵌入的本地关联记忆与召回内核：被反复使用的关系逐渐变强，长期闲置的关系自然衰减。

[中文](README.zh-CN.md) | [English](README.en.md) | [返回项目首页](README.md)

`TypeScript` · `Node >= 18` · `0 runtime deps` · `offline-friendly`

## 它解决什么问题

很多系统都需要回答同一个问题：**面对一个新输入，哪些过去的信息最值得再次使用？**

`pheromone_network` 将这个过程封装为一个小型本地组件。你提供向量，组件负责学习局部关联、给候选内容打分，并让长期不用的关联逐渐淡出。它不依赖远程模型、向量数据库或训练集，可以用于服务端、桌面端、边缘设备和各种 TypeScript 项目。

它可以作为独立的相似度召回层，也可以作为搜索、缓存、推荐、规则系统或智能体记忆的前置过滤器。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| **持续强化** | 通过局部 Hebbian 更新，让反复使用的关联获得更高影响力 |
| **自然淡忘** | 支持固定比例蒸发和按时间因子衰减，控制旧关系的影响范围 |
| **稀疏局部连接** | 每个输出只连接少量候选输入，降低干扰和计算成本 |
| **双时间尺度信息素** | 同时保留短期痕迹、长期记忆和固化状态 |
| **归一化召回** | 将查询和候选投影到联想空间，再用归一化余弦相似度排序 |
| **零运行时依赖** | 纯 TypeScript，仅使用 `node:crypto`，不要求联网或 GPU |

## 详细使用场景

### 1. 语义缓存

把请求、工具参数或内容生成提示编码成向量。新请求到来时，先从历史请求中找相似项；高分命中可以直接复用已有结果。每次复用后再次 `observe`，热门请求会更稳定，长期不用的缓存可以通过 `decayByFactor` 降权。

适合本地 API 缓存、工具调用缓存、内容生成缓存和重复查询优化。

### 2. 推荐与个性化排序

将用户点击、收藏、购买或常用操作表示为向量。行为发生时强化对应关联，展示候选内容时用 `score` 进行轻量重排。它可以放在已有推荐系统前面，负责捕捉短期兴趣和本地偏好。

适合常用功能排序、个性化菜单、轻量内容推荐和设备端偏好记忆。

### 3. 文档、工单和知识条目去重

新文档或工单写入前，先与历史向量打分。高分候选可以提示“可能已有相同问题”，并把已有解决方案放到前面。它不替代最终的去重规则，而是减少人工检查范围。

适合 FAQ 合并、客服工单归并、团队知识库维护和相似文档发现。

### 4. 日志与事件检索

把日志模板、错误码、服务名、时间窗口等信息组合成向量。查询故障现象时召回历史上相似的事件，再交给全文检索、规则引擎或人工排查。

适合故障排查、运行手册匹配、告警关联和异常上下文补全。

### 5. 离线、边缘和隐私敏感应用

所有关联都可以在本地维护，不需要把原始文本发送到远程服务。可以为每个设备、租户、项目或版本创建独立的 `RecallKernel`，自然隔离学习状态。

适合桌面工具、浏览器扩展、IoT、内网系统和不希望上传数据的场景。

### 6. 游戏、NPC 与模拟系统

将状态、地点、事件或玩家行为编码成向量。重复触发的状态关系会增强，NPC 或模拟实体可以优先选择曾经有效的行为路径；长时间不发生的关系会逐渐淡出。

适合 NPC 偏好、场景记忆、策略选择、动态难度和多实体模拟。

### 7. 作为检索前过滤层

面对大量候选时，先用本地内核取出一小批高分候选，再交给全文搜索、向量数据库或远程模型进行精排。这样可以减少后续检索范围、网络调用次数和上下文长度。

## 工作方式

```text
输入向量 -> 局部关联学习 -> 召回打分 -> 使用反馈强化
                                  ^
                         长期不用则逐渐衰减
```

最小接口只有三类：

```ts
kernel.observe(vector);                         // 写入并强化
kernel.score(queryVector, candidateVector);      // 召回打分
kernel.evaporate(rate);                         // 固定比例蒸发
kernel.decayByFactor(factor);                   // 按时间或业务周期衰减
```

## 30 秒上手

```ts
import { RecallKernel, ngramEmbed } from "pheromone_network";

const dim = 4096;
const kernel = new RecallKernel(dim, { codeDim: 512, seed: 1 });
const memories = ["客户偏好邮件沟通", "发票抬头是 Acme 科技", "结算周期是月结 30 天"];
const vectors = memories.map((text) => ngramEmbed(text, dim));

vectors.forEach((vector) => kernel.observe(vector));

const query = ngramEmbed("结算周期多久", dim);
const ranked = memories
  .map((text, index) => ({ text, score: kernel.score(query, vectors[index]) }))
  .sort((left, right) => right.score - left.score);

console.log(ranked[0]);
// { text: "结算周期是月结 30 天", score: ... }
```

## 接入方式

### 使用已有向量

`RecallKernel` 接受 `number[]`。如果项目已经有 embedding、哈希特征或模型输出，可以直接传入，不需要更换上层数据结构。

### 使用内置编码器

`ngramEmbed(text, dim)` 提供确定性的字符 2-5 gram 编码，适合零依赖原型、短文本、日志模板和离线场景。对复杂语义或多语言长文本，可以替换为项目已有的 embedding 函数。

### 按范围隔离状态

按租户、用户、项目、设备或版本创建独立内核，避免不同范围之间互相强化。需要持久化时，可以保存业务侧的原始记录和向量，并在进程启动时重新观察；当前 SDK 本身不绑定数据库格式。

## 什么时候不适合

它是关联学习与候选召回组件，不是全文搜索引擎、关系数据库、向量数据库或通用分类器。需要精确关键词过滤、复杂结构化查询、海量持久化索引或严格监督学习时，应与对应系统组合使用。

## 核心 API

- 引擎：`LocalPheromoneNetwork`、`LocalPheromoneLayer`、`LocalLearningConfig`
- 召回：`RecallKernel`、`normalizedCosine`、`DEFAULT_CODE_DIM`
- 编码：`ngramEmbed`、`cosine`、`Embedder`

## 项目结构

| 文件 | 职责 |
| --- | --- |
| `src/pheromone.ts` | 稀疏局部层、手工 Hebbian 更新和双尺度信息素 |
| `src/recall.ts` | `RecallKernel`、折叠、联想投影和归一化余弦打分 |
| `src/embed.ts` | 确定性字符 2-5 gram 编码器与 `cosine` |
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
