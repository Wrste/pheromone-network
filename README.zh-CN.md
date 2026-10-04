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

## 现在的智能体可以用它做什么

在一个典型的 Agent 流程里，模型负责理解和生成，`pheromone_network` 负责从历史信息中找出“此刻最值得给模型看的内容”。它可以接在上下文窗口之前，作为一个会根据使用反馈变化的本地记忆层：

- **长期记忆召回**：保存用户偏好、项目约定、历史决定和重要事实，在新对话中只注入相关内容。
- **会话上下文筛选**：从很长的聊天记录、任务记录或观察结果中筛选少量高分片段，控制上下文长度和成本。
- **工具结果复用**：记录搜索、数据库查询、代码分析或 API 调用的结果；遇到相似任务时优先找到过去可复用的结果。
- **任务经验积累**：把“问题、采取的步骤、最终结果”编码后写入；相似任务出现时召回过去有效的处理路径。
- **用户偏好维护**：强化用户反复确认或使用的表达方式、输出格式和操作习惯，让 Agent 的响应更贴合个人工作流。
- **反思与失败记录**：把失败原因、修正方案和验证结果作为候选记忆保存，后续遇到相似错误时提醒模型避开旧路径。
- **多 Agent 共享记忆**：为团队、项目或工作区维护独立内核，让不同 Agent 共享经过使用验证的事实和经验。

一个简单的接入顺序是：**用户输入和任务状态 -> 生成查询向量 -> 召回 Top-K 记忆 -> 放入模型上下文 -> 根据实际使用结果再次强化**。它不替代模型，也不负责生成答案，而是让模型每次都能更快拿到相关的历史信息。

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

## API 接口参考

### `RecallKernel`

需要在线记忆、候选排序或本地召回层时，优先使用这个高层 API。

```ts
new RecallKernel(dim: number, options?: RecallKernelOptions)
```

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `codeDim` | `512` | 内部联想维度，不超过 `dim` |
| `maxNeighbors` | `8` | 每个输出的最大局部连接数 |
| `tagCount` | `32` | 拓扑标签数量 |
| `tagDistance` | `1` | 允许连接的最大标签距离 |
| `config` | 库默认配置 | 部分 `LocalLearningConfig` 覆盖项 |
| `seed` | `1` | 确定性初始化；设为 `null` 使用随机初始化 |

| 方法或属性 | 返回值 | 用途 |
| --- | --- | --- |
| `observe(vector)` | `TrainStepReport` | 通过自联想学习一条记忆向量 |
| `score(query, candidate)` | `number` | 归一化相似度，越高越接近 |
| `encode(vector)` | `number[]` | 将向量投影到已学习的联想空间 |
| `evaporate(rate?)` | `number` | 按 `[0, 1]` 的比率衰减信息素，返回受影响连接数 |
| `decayByFactor(factor)` | `number` | 将信息素乘以小于 `1` 的因子 |
| `pheromoneMass()` | `number` | 统计有效连接的长期信息素总量 |
| `trained` | `boolean` | 是否至少观察过一条向量 |
| `trainSteps` | `number` | 观察次数 |
| `loss` | `number \| null` | 最近一次观察的损失 |
| `dim`、`codeDim`、`network` | 属性 | 输入维度、内部维度和底层网络 |

### 在智能体中接入已有 embedding

内核接受任意 `number[]`，因此可以直接复用项目已有的 embedding 服务，并将原文和向量一起保存：

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

将 `existingEmbeddingModel.embed` 替换为你的 embedding 函数即可。传给 `RecallKernel` 的 `dim` 必须与向量维度一致。

### `LocalPheromoneNetwork`

需要明确控制输入/输出维度、隐藏层、激活函数、标签、拓扑或监督式局部训练时，使用底层网络：

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

`input` 和 `target` 是按行组织的矩阵，行数必须一致；每行输入有 `inputSize` 个值，每行目标有 `outputSize` 个值。`forwardWithCache` 还会返回隐藏层激活值和激活前值。`localTrainStep` 返回包含 `loss`、`previousLoss`、`improved`、`mode`、`budgetPerOutput`、`activeSynapses` 和 `prediction` 的 `TrainStepReport`。传入 `outputMask` 可以对输出维度加权或禁用训练。

网络选项包括 `maxNeighbors`、`tagCount`、`tagDistance`、`connectionRadius`、`activation`、`inputShape`、`hiddenShapes`、`outputShape`、`inputTags`、`outputTags`、`config` 和 `seed`。布局支持一维/二维；`inputTags` 和 `outputTags` 的长度必须与对应层大小一致，`hiddenShapes` 的长度必须与 `hiddenSizes` 一致。`config` 接受部分 `LocalLearningConfig`，并与 `DEFAULT_LOCAL_LEARNING_CONFIG` 合并。

### `LocalPheromoneLayer`

需要自定义拓扑或检查局部连接时，可以直接创建 `new LocalPheromoneLayer(inFeatures, outFeatures, options?)`。主要方法如下：

| 方法 | 用途 |
| --- | --- |
| `forward(matrix)` | 计算层输出 |
| `feedbackToInput(matrix)` | 将输出信号传回输入特征 |
| `localUpdate(pre, post, localError, options)` | 执行手工局部 Hebbian 更新 |
| `connectedInputs(outputIndex)` | 查看某个输出连接的输入索引 |
| `effectivePheromone()` | 读取短期/长期信息素混合值 |
| `pheromoneGate()` | 读取归一化信息素门控 |
| `effectiveWeight()` | 读取经过信息素门控后的权重 |

层选项支持 `maxNeighbors`、`connectionRadius`、`tagCount`、`tagDistance`、`inputShape`、`outputShape`、`inputTags`、`outputTags`、`usePheromoneGate`、`shortPheromoneWeight`、`longPheromoneWeight`、`bias` 和 `seed`。层对象公开的类型化数组是实时状态，只有在明确管理引擎状态时才应直接修改。

可检查的层状态包括 `inFeatures`、`outFeatures`、`maxNeighbors`、`connectionRadius`、`tagCount`、`tagDistance`、`usePheromoneGate`、`neighborIndices`、`connectionMask`、`inputPositions`、`outputPositions`、`weight`、`bias`、`pheromone`、`shortPheromone`、`consolidation`、`lastUpdateMask`、`inputDegree` 和 `lastDirectionalDerivative`。

### 向量工具与公共类型

| 导出项 | 接口约定 |
| --- | --- |
| `ngramEmbed(text, dim?)` | 确定性的字符 2-5 gram 向量，L2 归一化；默认维度 `4096` |
| `cosine(left, right)` | 等长归一化向量的点积；长度不同返回 `0` |
| `normalizedCosine(left, right)` | 适用于任意模长向量的真正余弦相似度；长度不同时按双方共有的前缀计算 |
| `Matrix` | `number[][]`，按行表示特征 |
| `Embedder` | `(text: string) => number[]` |
| `ActivationName` | `"tanh" \| "relu" \| "sigmoid" \| "identity"` |
| `DEFAULT_CODE_DIM` | `512` |
| `DEFAULT_RECALL_KERNEL_OPTIONS` | `RecallKernel` 的默认拓扑和随机种子 |
| `DEFAULT_LOCAL_LEARNING_CONFIG` | 完整的默认训练和信息素配置 |

`LocalLearningConfig` 控制学习率、自适应预算、蒸发、信号/权重裁剪、短期/长期信息素权重和固化，字段分为三组：

- **学习与预算**：`learningRate`、`minBudgetPerOutput`、`maxBudgetPerOutput`、`initialBudgetPerOutput`、`shrinkFactor`、`growFactor`、`lossTolerance`。
- **信息素与稳定性**：`evaporation`、`pheromoneReinforcement`、`synapseDecay`、`signalClip`、`weightClip`、`neighborFollowDistance`、`minPheromone`、`maxPheromone`。
- **时间尺度与固化**：`shortPheromoneEvaporation`、`longPheromoneEvaporation`、`shortPheromoneReinforcement`、`longPheromoneReinforcement`、`shortPheromoneWeight`、`longPheromoneWeight`、`consolidationStrength`、`consolidationGrowth`、`consolidationDecay`、`consolidationThreshold`、`consolidationLrFloor`、`consolidationLossGate`。

对于高级方法 `LocalPheromoneLayer.localUpdate`，`LocalUpdateOptions` 必须提供 `budgetPerOutput`、`mode`、`config` 和 `consolidate`。

选项接口包括 `RecallKernelOptions`、`LocalPheromoneNetworkOptions`、`LocalPheromoneLayerOptions` 和 `LocalUpdateOptions`。网络对象还公开 `inputSize`、`hiddenSizes`、`outputSize`、`activation`、`maxNeighbors`、`config`、`layers`、`budgetPerOutput` 和 `previousLoss`，可用于检查和监控。

## 接入方式

### 使用已有向量

`RecallKernel` 接受 `number[]`。如果项目已经有 embedding、哈希特征或模型输出，可以直接传入，不需要更换上层数据结构。

### 使用内置编码器

`ngramEmbed(text, dim)` 提供确定性的字符 2-5 gram 编码，适合零依赖原型、短文本、日志模板和离线场景。对复杂语义或多语言长文本，可以替换为项目已有的 embedding 函数。

### 按范围隔离状态

按租户、用户、项目、设备或版本创建独立内核，避免不同范围之间互相强化。需要持久化时，可以保存业务侧的原始记录和向量，并在进程启动时重新观察；当前 SDK 本身不绑定数据库格式。

## 什么时候不适合

它是关联学习与候选召回组件，不是全文搜索引擎、关系数据库、向量数据库或通用分类器。需要精确关键词过滤、复杂结构化查询、海量持久化索引或严格监督学习时，应与对应系统组合使用。

## 场景化测试结果

可复现用例位于 [`test/scenarios.test.ts`](test/scenarios.test.ts)。每个检索场景有 3 个候选，各写入一次；使用 `RecallKernel(256, { codeDim: 128, seed: 17 })` 和内置 `ngramEmbed`。四个场景的预期候选均排在首位：

| 场景 | 预期首位结果 | 首位分数 | 次位分数 | 结果 |
| --- | --- | ---: | ---: | --- |
| 工具缓存 | 匹配的 `GET /api/orders/42` 请求 | 0.9701 | 0.7412 | 通过 |
| 智能体记忆 | 简洁英文摘要的用户偏好 | 0.9106 | 0.4990 | 通过 |
| 客服工单 | 密码重置邮件链接过期 | 0.8025 | 0.5011 | 通过 |
| 故障日志 | 支付网关 504 超时 | 0.8433 | 0.5746 | 通过 |

维护场景也通过：两次写入后调用 `decayByFactor(0.5)`，信息素总量从 **984.6413** 降到 **492.3206**。加上原有 7 项单元测试，`npm test` 共 **12/12 通过**。

运行 `npm test` 可复现并查看分数。这些是小规模、确定性的近似文本匹配与衰减样例，不代表真实数据集上的召回率。内置字符 n-gram 编码器不能保证识别不同措辞的语义等价；需要语义检索时应接入合适的 embedding 模型。上文提到的推荐、边缘部署和游戏场景尚未由本测试集测量。

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
npm install pheromone_network
```

本地开发时运行：

```bash
npm install
npm run build
npm test
```

## 实现来源

算法实现参考论文 **Local Pheromone Network: Sparse Local Learning with Multi-Scale Synaptic Trails, Consolidation, and Replay**（arXiv:2606.30669）。本 SDK 是该论文所述方法的零依赖 TypeScript 核心子集移植。

- 论文：<https://arxiv.org/abs/2606.30669>
- DOI：<https://doi.org/10.48550/arXiv.2606.30669>

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。
