/**
 * 召回内核：把信息素内核（LocalPheromoneNetwork）用作稀疏局部自联想投影。
 *
 * 设计要点：
 * - 每个记忆分区（scope.key()）持有一个 RecallKernel，只在本分区内学习，天然满足租户/版本隔离。
 * - 写入时以 key 自身为目标做一次局部 Hebbian 学习（自联想），被反复引用的稀疏连接会被信息素强化。
 * - 召回时用内核前向把 query 与候选 key 投影到同一"联想空间"，再用**归一化余弦**打分。
 *   注意：ngramEmbed 已 L2 归一化，所以旧打分用的 cosine 实为点积；内核输出未归一化，
 *   必须显式归一化，否则会被向量模长带偏。
 * - 内核维度通过确定性折叠把 4096 维 embedding 压到 codeDim（默认 512）：折叠覆盖全部输入维度，
 *   使 LocalPheromoneLayer 的构图成本从 O(dim^2) 降到 O(codeDim^2)，写入/召回热路径保持轻量。
 */

import {
  LocalPheromoneNetwork,
  type LocalLearningConfig,
  type Matrix,
  type TrainStepReport,
} from "./pheromone";

export interface RecallKernelOptions {
  maxNeighbors?: number;
  tagCount?: number;
  tagDistance?: number;
  codeDim?: number;
  config?: Partial<LocalLearningConfig>;
  seed?: number | null;
}

export const DEFAULT_CODE_DIM = 512;

export const DEFAULT_RECALL_KERNEL_OPTIONS = {
  maxNeighbors: 8,
  tagCount: 32,
  tagDistance: 1,
  codeDim: DEFAULT_CODE_DIM,
  seed: 1,
};

/**
 * 真正的余弦：对未归一化向量同样成立（旧 cosine 只在已归一化时等价于点积）。
 */
export function normalizedCosine(left: number[], right: number[]): number {
  const length = Math.min(left.length, right.length);
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let i = 0; i < length; i += 1) {
    dot += left[i] * right[i];
    leftNorm += left[i] * left[i];
    rightNorm += right[i] * right[i];
  }
  const denom = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
  return denom > 0 ? dot / denom : 0;
}

export class RecallKernel {
  readonly dim: number;
  readonly codeDim: number;
  readonly network: LocalPheromoneNetwork;
  private observations = 0;
  private lastLoss: number | null = null;

  constructor(dim: number, options: RecallKernelOptions = {}) {
    if (dim <= 0) {
      throw new Error("dim must be positive");
    }
    this.dim = Math.trunc(dim);
    const requestedCodeDim = options.codeDim ?? DEFAULT_RECALL_KERNEL_OPTIONS.codeDim;
    this.codeDim = Math.max(1, Math.min(this.dim, requestedCodeDim));
    this.network = new LocalPheromoneNetwork(this.codeDim, [], this.codeDim, {
      maxNeighbors: options.maxNeighbors ?? DEFAULT_RECALL_KERNEL_OPTIONS.maxNeighbors,
      tagCount: options.tagCount ?? DEFAULT_RECALL_KERNEL_OPTIONS.tagCount,
      tagDistance: options.tagDistance ?? DEFAULT_RECALL_KERNEL_OPTIONS.tagDistance,
      config: options.config,
      seed: options.seed ?? DEFAULT_RECALL_KERNEL_OPTIONS.seed,
    });
  }

  get trained(): boolean {
    return this.observations > 0;
  }

  get trainSteps(): number {
    return this.observations;
  }

  get loss(): number | null {
    return this.lastLoss;
  }

  /** 确定性折叠：把高维 embedding 压到 codeDim，覆盖全部输入维度。 */
  private fold(vector: number[]): number[] {
    const out = new Array<number>(this.codeDim).fill(0);
    const length = Math.min(vector.length, this.dim);
    for (let i = 0; i < length; i += 1) {
      out[i % this.codeDim] += vector[i];
    }
    return out;
  }

  /** 内核前向：把 embedding 折叠后投影到学习到的联想空间。 */
  encode(vector: number[]): number[] {
    return this.network.forward([this.fold(vector)])[0];
  }

  /** 自联想学习：以向量自身为目标，强化当前触发的稀疏连接与信息素。 */
  observe(vector: number[]): TrainStepReport {
    const folded = this.fold(vector);
    const target: Matrix = [folded];
    const report = this.network.localTrainStep([folded], target);
    this.observations += 1;
    this.lastLoss = report.loss;
    return report;
  }

  /** 内核打分：两侧都过前向后再做归一化余弦。 */
  score(queryVector: number[], keyVector: number[]): number {
    return normalizedCosine(this.encode(queryVector), this.encode(keyVector));
  }

  /**
   * 固定比率蒸发信息素：长期未被强化（未被选中）的稀疏连接按比例衰减，下限 minPheromone。
   * 与 localUpdate 内的蒸发叠加，构成"写入强化 / 静置弱化"的双向动力学。
   */
  evaporate(rate?: number): number {
    const configured = rate ?? this.network.config.evaporation;
    const clamped = Math.min(Math.max(configured, 0), 1);
    return this.decayByFactor(1 - clamped);
  }

  /** 按乘法因子衰减信息素（factor∈[0,1]；factor>=1 视为不衰减）。供时间衰减复用。 */
  decayByFactor(factor: number): number {
    if (!(factor < 1)) {
      return 0;
    }
    const floor = this.network.config.minPheromone;
    let affected = 0;
    for (const layer of this.network.layers) {
      for (let i = 0; i < layer.pheromone.length; i += 1) {
        if (!layer.connectionMask[i]) {
          continue;
        }
        if (layer.pheromone[i] > floor) {
          layer.pheromone[i] = Math.max(layer.pheromone[i] * factor, floor);
          affected += 1;
        }
        if (layer.shortPheromone[i] > floor) {
          layer.shortPheromone[i] = Math.max(layer.shortPheromone[i] * factor, floor);
        }
      }
    }
    return affected;
  }

  /** 当前信息素总量（只计有效连接），用于观测蒸发/强化效果。 */
  pheromoneMass(): number {
    let sum = 0;
    for (const layer of this.network.layers) {
      for (let i = 0; i < layer.pheromone.length; i += 1) {
        if (layer.connectionMask[i]) {
          sum += layer.pheromone[i];
        }
      }
    }
    return sum;
  }
}
