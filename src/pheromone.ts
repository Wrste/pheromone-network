/**
 * Dependency-free TypeScript port of the local pheromone kernel core subset.
 *
 * Ported from local_pheromone_network.py: sparse tag-restricted local layers with
 * a manual Hebbian-style update, dual-timescale pheromone traces and consolidation.
 * Forward-only (no autograd). Structural plasticity, replay and GELU are intentionally
 * omitted from this core subset.
 */

export type Matrix = number[][];

export type ActivationName = "tanh" | "relu" | "sigmoid" | "identity";

export interface LocalLearningConfig {
  learningRate: number;
  minBudgetPerOutput: number;
  maxBudgetPerOutput: number | null;
  initialBudgetPerOutput: number | null;
  shrinkFactor: number;
  growFactor: number;
  lossTolerance: number;
  evaporation: number;
  pheromoneReinforcement: number;
  synapseDecay: number;
  signalClip: number;
  weightClip: number;
  neighborFollowDistance: number;
  minPheromone: number;
  maxPheromone: number;
  shortPheromoneEvaporation: number | null;
  longPheromoneEvaporation: number | null;
  shortPheromoneReinforcement: number | null;
  longPheromoneReinforcement: number | null;
  shortPheromoneWeight: number;
  longPheromoneWeight: number;
  consolidationStrength: number;
  consolidationGrowth: number;
  consolidationDecay: number;
  consolidationThreshold: number;
  consolidationLrFloor: number;
  consolidationLossGate: number | null;
}

export const DEFAULT_LOCAL_LEARNING_CONFIG: LocalLearningConfig = {
  learningRate: 0.05,
  minBudgetPerOutput: 1,
  maxBudgetPerOutput: null,
  initialBudgetPerOutput: null,
  shrinkFactor: 0.7,
  growFactor: 1.5,
  lossTolerance: 1e-7,
  evaporation: 0.02,
  pheromoneReinforcement: 0.05,
  synapseDecay: 0.001,
  signalClip: 1.0,
  weightClip: 5.0,
  neighborFollowDistance: 0.18,
  minPheromone: 0.01,
  maxPheromone: 8.0,
  shortPheromoneEvaporation: null,
  longPheromoneEvaporation: null,
  shortPheromoneReinforcement: null,
  longPheromoneReinforcement: null,
  shortPheromoneWeight: 0.0,
  longPheromoneWeight: 1.0,
  consolidationStrength: 0.0,
  consolidationGrowth: 0.03,
  consolidationDecay: 0.0005,
  consolidationThreshold: 1.2,
  consolidationLrFloor: 0.05,
  consolidationLossGate: null,
};

export interface LocalPheromoneLayerOptions {
  maxNeighbors?: number;
  connectionRadius?: number | null;
  tagCount?: number;
  tagDistance?: number;
  inputShape?: number[] | null;
  outputShape?: number[] | null;
  inputTags?: number[] | null;
  outputTags?: number[] | null;
  usePheromoneGate?: boolean;
  shortPheromoneWeight?: number;
  longPheromoneWeight?: number;
  bias?: boolean;
  seed?: number | null;
}

export interface LocalUpdateOptions {
  budgetPerOutput: number;
  mode: string;
  config: LocalLearningConfig;
  consolidate: boolean;
}

export interface LocalPheromoneNetworkOptions {
  maxNeighbors?: number;
  tagCount?: number;
  tagDistance?: number;
  connectionRadius?: number | null;
  activation?: ActivationName;
  inputShape?: number[] | null;
  hiddenShapes?: (number[] | null)[] | null;
  outputShape?: number[] | null;
  inputTags?: number[] | null;
  outputTags?: number[] | null;
  config?: Partial<LocalLearningConfig>;
  seed?: number | null;
}

export interface TrainStepReport {
  loss: number;
  previousLoss: number | null;
  improved: boolean | null;
  mode: string;
  budgetPerOutput: number;
  activeSynapses: number;
  prediction: Matrix;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGaussian(seed: number): () => number {
  const random = mulberry32(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u1 = 0;
    while (u1 <= 1e-12) {
      u1 = random();
    }
    const u2 = random();
    const magnitude = Math.sqrt(-2 * Math.log(u1));
    spare = magnitude * Math.sin(2 * Math.PI * u2);
    return magnitude * Math.cos(2 * Math.PI * u2);
  };
}

function zeros(rows: number, cols: number): Matrix {
  const out: Matrix = [];
  for (let r = 0; r < rows; r += 1) {
    out.push(new Array<number>(cols).fill(0));
  }
  return out;
}

function euclidean(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function gridPositions(count: number, shape: number[] | null): Matrix {
  if (count <= 0) {
    throw new Error("count must be positive");
  }
  if (shape === null || shape.length === 1) {
    if (count === 1) {
      return [[0, 0]];
    }
    const positions: Matrix = [];
    for (let i = 0; i < count; i += 1) {
      positions.push([i / (count - 1), 0]);
    }
    return positions;
  }
  if (shape.length !== 2) {
    throw new Error("only 1D or 2D unit layouts are supported");
  }
  const rows = shape[0];
  const cols = shape[1];
  const rowDen = Math.max(rows - 1, 1);
  const colDen = Math.max(cols - 1, 1);
  const positions: Matrix = [];
  for (let i = 0; i < rows * cols; i += 1) {
    const row = Math.floor(i / cols);
    positions.push([row / rowDen, (i % cols) / colDen]);
  }
  return positions;
}

function defaultTags(count: number, tagCount: number, shape: number[] | null): number[] {
  if (tagCount <= 0) {
    return new Array<number>(count).fill(0);
  }
  if (shape !== null && shape.length === 2) {
    const rows = shape[0];
    const cols = shape[1];
    const tags: number[] = [];
    for (let i = 0; i < rows * cols; i += 1) {
      tags.push(Math.floor((Math.floor(i / cols) * tagCount) / Math.max(rows, 1)));
    }
    return tags;
  }
  const tags: number[] = [];
  for (let i = 0; i < count; i += 1) {
    tags.push(Math.floor((i * tagCount) / Math.max(count, 1)));
  }
  return tags;
}

function resolveTags(
  tags: number[] | null,
  count: number,
  tagCount: number,
  shape: number[] | null,
): number[] {
  if (tags === null) {
    return defaultTags(count, tagCount, shape);
  }
  if (tags.length !== count) {
    throw new Error(`expected ${count} tags, got ${tags.length}`);
  }
  return tags.slice();
}

export class LocalPheromoneLayer {
  readonly inFeatures: number;
  readonly outFeatures: number;
  readonly maxNeighbors: number;
  readonly connectionRadius: number | null;
  readonly tagCount: number;
  readonly tagDistance: number;
  readonly usePheromoneGate: boolean;
  readonly shortPheromoneWeight: number;
  readonly longPheromoneWeight: number;
  readonly neighborIndices: Int32Array;
  readonly connectionMask: Uint8Array;
  readonly inputPositions: Matrix;
  readonly outputPositions: Matrix;
  readonly weight: Float64Array;
  readonly bias: Float64Array | null;
  readonly pheromone: Float64Array;
  readonly shortPheromone: Float64Array;
  readonly consolidation: Float64Array;
  readonly lastUpdateMask: Uint8Array;
  readonly inputDegree: Float64Array;
  lastDirectionalDerivative = 0;

  constructor(inFeatures: number, outFeatures: number, options: LocalPheromoneLayerOptions = {}) {
    if (inFeatures <= 0 || outFeatures <= 0) {
      throw new Error("inFeatures and outFeatures must be positive");
    }
    const maxNeighborsOption = options.maxNeighbors ?? 8;
    if (maxNeighborsOption <= 0) {
      throw new Error("maxNeighbors must be positive");
    }

    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.maxNeighbors = Math.min(maxNeighborsOption, inFeatures);
    this.connectionRadius = options.connectionRadius ?? null;
    this.tagCount = options.tagCount ?? 4;
    this.tagDistance = options.tagDistance ?? 0;
    this.usePheromoneGate = options.usePheromoneGate ?? true;
    this.shortPheromoneWeight = options.shortPheromoneWeight ?? 0.0;
    this.longPheromoneWeight = options.longPheromoneWeight ?? 1.0;

    const inputShape = options.inputShape ?? null;
    const outputShape = options.outputShape ?? null;
    const inputPositions = gridPositions(inFeatures, inputShape);
    const outputPositions = gridPositions(outFeatures, outputShape);
    const inputTags = resolveTags(options.inputTags ?? null, inFeatures, this.tagCount, inputShape);
    const outputTags = resolveTags(options.outputTags ?? null, outFeatures, this.tagCount, outputShape);

    this.inputPositions = inputPositions;
    this.outputPositions = outputPositions;

    const size = outFeatures * this.maxNeighbors;
    const neighborIndices = new Int32Array(size);
    const connectionMask = new Uint8Array(size);

    for (let outIdx = 0; outIdx < outFeatures; outIdx += 1) {
      const base = outIdx * this.maxNeighbors;
      let candidates: number[] = [];
      for (let inIdx = 0; inIdx < inFeatures; inIdx += 1) {
        const tagAllowed = Math.abs(outputTags[outIdx] - inputTags[inIdx]) <= this.tagDistance;
        if (!tagAllowed) {
          continue;
        }
        if (this.connectionRadius !== null) {
          const distance = euclidean(outputPositions[outIdx], inputPositions[inIdx]);
          if (distance > this.connectionRadius) {
            continue;
          }
        }
        candidates.push(inIdx);
      }
      if (candidates.length === 0) {
        for (let inIdx = 0; inIdx < inFeatures; inIdx += 1) {
          if (Math.abs(outputTags[outIdx] - inputTags[inIdx]) <= this.tagDistance) {
            candidates.push(inIdx);
          }
        }
      }
      if (candidates.length === 0) {
        throw new Error(
          "output unit has no tag-compatible input candidates; adjust tagCount, tagDistance, inputTags, or outputTags",
        );
      }
      candidates.sort(
        (left, right) =>
          euclidean(outputPositions[outIdx], inputPositions[left]) -
          euclidean(outputPositions[outIdx], inputPositions[right]),
      );
      const take = Math.min(this.maxNeighbors, candidates.length);
      for (let slot = 0; slot < take; slot += 1) {
        neighborIndices[base + slot] = candidates[slot];
        connectionMask[base + slot] = 1;
      }
    }

    this.neighborIndices = neighborIndices;
    this.connectionMask = connectionMask;

    const seed = options.seed ?? Math.floor(Math.random() * 0xffffffff);
    const gaussian = makeGaussian(seed);
    const scale = 1 / Math.sqrt(Math.max(this.maxNeighbors, 1));
    const weight = new Float64Array(size);
    for (let i = 0; i < size; i += 1) {
      if (connectionMask[i]) {
        weight[i] = gaussian() * scale;
      }
    }
    this.weight = weight;

    this.bias = options.bias === false ? null : new Float64Array(outFeatures);

    const pheromone = new Float64Array(size);
    const shortPheromone = new Float64Array(size);
    const consolidation = new Float64Array(size);
    for (let i = 0; i < size; i += 1) {
      if (connectionMask[i]) {
        pheromone[i] = 1;
        shortPheromone[i] = 1;
      }
    }
    this.pheromone = pheromone;
    this.shortPheromone = shortPheromone;
    this.consolidation = consolidation;
    this.lastUpdateMask = new Uint8Array(size);

    const inputDegree = new Float64Array(inFeatures);
    for (let i = 0; i < size; i += 1) {
      if (connectionMask[i]) {
        inputDegree[neighborIndices[i]] += 1;
      }
    }
    for (let i = 0; i < inFeatures; i += 1) {
      if (inputDegree[i] < 1) {
        inputDegree[i] = 1;
      }
    }
    this.inputDegree = inputDegree;
  }

  effectivePheromone(): Float64Array {
    const size = this.neighborIndices.length;
    const out = new Float64Array(size);
    const totalWeight = this.shortPheromoneWeight + this.longPheromoneWeight;
    if (totalWeight <= 0) {
      out.set(this.pheromone);
      return out;
    }
    for (let i = 0; i < size; i += 1) {
      out[i] =
        (this.shortPheromone[i] * this.shortPheromoneWeight +
          this.pheromone[i] * this.longPheromoneWeight) /
        totalWeight;
    }
    return out;
  }

  pheromoneGate(): Float64Array {
    const size = this.neighborIndices.length;
    const gate = new Float64Array(size);
    if (!this.usePheromoneGate) {
      gate.fill(1);
      return gate;
    }
    const effective = this.effectivePheromone();
    const { outFeatures, maxNeighbors, connectionMask } = this;
    for (let outIdx = 0; outIdx < outFeatures; outIdx += 1) {
      const base = outIdx * maxNeighbors;
      let validCount = 0;
      let trailMass = 0;
      for (let slot = 0; slot < maxNeighbors; slot += 1) {
        if (connectionMask[base + slot]) {
          validCount += 1;
        }
        const value = connectionMask[base + slot] ? Math.max(effective[base + slot], 0) : 0;
        gate[base + slot] = value;
        trailMass += value;
      }
      if (validCount < 1) {
        validCount = 1;
      }
      if (trailMass < 1e-6) {
        trailMass = 1e-6;
      }
      for (let slot = 0; slot < maxNeighbors; slot += 1) {
        const normalized = (gate[base + slot] / trailMass) * validCount;
        const clamped = Math.min(Math.max(normalized, 0), 2);
        gate[base + slot] = 0.5 + 0.5 * clamped;
      }
    }
    return gate;
  }

  effectiveWeight(): Float64Array {
    const size = this.neighborIndices.length;
    const out = new Float64Array(size);
    const gate = this.pheromoneGate();
    for (let i = 0; i < size; i += 1) {
      out[i] = this.weight[i] * this.connectionMask[i] * gate[i];
    }
    return out;
  }

  connectedInputs(outputIndex: number): number[] {
    if (outputIndex < 0 || outputIndex >= this.outFeatures) {
      throw new Error("outputIndex out of range");
    }
    const base = outputIndex * this.maxNeighbors;
    const connected: number[] = [];
    for (let slot = 0; slot < this.maxNeighbors; slot += 1) {
      if (this.connectionMask[base + slot]) {
        connected.push(this.neighborIndices[base + slot]);
      }
    }
    return connected;
  }

  forward(x: Matrix): Matrix {
    const rows = x.length;
    const out = zeros(rows, this.outFeatures);
    const effective = this.effectiveWeight();
    const { outFeatures, maxNeighbors, neighborIndices, connectionMask, bias } = this;
    for (let row = 0; row < rows; row += 1) {
      const values = x[row];
      const target = out[row];
      for (let outIdx = 0; outIdx < outFeatures; outIdx += 1) {
        const base = outIdx * maxNeighbors;
        let acc = 0;
        for (let slot = 0; slot < maxNeighbors; slot += 1) {
          if (connectionMask[base + slot]) {
            acc += values[neighborIndices[base + slot]] * effective[base + slot];
          }
        }
        if (bias !== null) {
          acc += bias[outIdx];
        }
        target[outIdx] = acc;
      }
    }
    return out;
  }

  feedbackToInput(postSignal: Matrix): Matrix {
    const rows = postSignal.length;
    const out = zeros(rows, this.inFeatures);
    const effective = this.effectiveWeight();
    const { outFeatures, maxNeighbors, neighborIndices, connectionMask, inputDegree } = this;
    for (let row = 0; row < rows; row += 1) {
      const signal = postSignal[row];
      const target = out[row];
      for (let outIdx = 0; outIdx < outFeatures; outIdx += 1) {
        const base = outIdx * maxNeighbors;
        const value = signal[outIdx];
        for (let slot = 0; slot < maxNeighbors; slot += 1) {
          if (connectionMask[base + slot]) {
            target[neighborIndices[base + slot]] += value * effective[base + slot];
          }
        }
      }
      for (let inIdx = 0; inIdx < this.inFeatures; inIdx += 1) {
        target[inIdx] /= Math.sqrt(inputDegree[inIdx]);
      }
    }
    return out;
  }

  localUpdate(pre: Matrix, post: Matrix, localError: Matrix, options: LocalUpdateOptions): number {
    const config = options.config;
    const { outFeatures, maxNeighbors, neighborIndices, connectionMask, bias } = this;
    const size = neighborIndices.length;
    const rows = pre.length;

    const signal = new Float64Array(size);
    const cofire = new Float64Array(size);
    for (let row = 0; row < rows; row += 1) {
      const preRow = pre[row];
      const postRow = post[row];
      const errorRow = localError[row];
      for (let outIdx = 0; outIdx < outFeatures; outIdx += 1) {
        const base = outIdx * maxNeighbors;
        const errorValue = errorRow[outIdx];
        const postValue = postRow[outIdx];
        for (let slot = 0; slot < maxNeighbors; slot += 1) {
          if (!connectionMask[base + slot]) {
            continue;
          }
          const localPre = preRow[neighborIndices[base + slot]];
          signal[base + slot] += errorValue * localPre;
          cofire[base + slot] += Math.abs(localPre) * Math.abs(postValue);
        }
      }
    }

    const invRows = rows > 0 ? 1 / rows : 0;
    const gate = this.pheromoneGate();
    for (let i = 0; i < size; i += 1) {
      let value = signal[i] * invRows * gate[i];
      value = Math.min(Math.max(value, -config.signalClip), config.signalClip);
      signal[i] = connectionMask[i] ? value : 0;
      cofire[i] = connectionMask[i] ? cofire[i] * invRows : 0;
    }

    const selected = this.selectSynapses(
      signal,
      options.budgetPerOutput,
      options.mode,
      config.neighborFollowDistance,
    );
    const plasticity = this.plasticityScale(config);

    const candidateDirection = new Float64Array(size);
    for (let i = 0; i < size; i += 1) {
      candidateDirection[i] = signal[i] * selected[i] * plasticity[i];
    }

    let directionalDerivative = 0;
    for (let i = 0; i < size; i += 1) {
      directionalDerivative += -signal[i] * candidateDirection[i];
    }

    let biasDirection: Float64Array | null = null;
    if (bias !== null) {
      biasDirection = new Float64Array(outFeatures);
      for (let outIdx = 0; outIdx < outFeatures; outIdx += 1) {
        let acc = 0;
        for (let row = 0; row < rows; row += 1) {
          acc += localError[row][outIdx];
        }
        acc = rows > 0 ? acc / rows : 0;
        biasDirection[outIdx] = Math.min(Math.max(acc, -config.signalClip), config.signalClip);
      }
      for (let outIdx = 0; outIdx < outFeatures; outIdx += 1) {
        directionalDerivative += -biasDirection[outIdx] * biasDirection[outIdx];
      }
    }

    this.lastDirectionalDerivative = directionalDerivative;
    const isDescent = Number.isFinite(directionalDerivative) && directionalDerivative < 0;

    const accepted = new Uint8Array(size);
    let acceptedCount = 0;
    if (isDescent) {
      for (let i = 0; i < size; i += 1) {
        if (selected[i]) {
          accepted[i] = 1;
          acceptedCount += 1;
        }
      }
    }

    if (acceptedCount > 0) {
      for (let i = 0; i < size; i += 1) {
        this.weight[i] += config.learningRate * candidateDirection[i];
      }
      for (let i = 0; i < size; i += 1) {
        if (connectionMask[i] && !accepted[i]) {
          this.weight[i] *= 1 - config.synapseDecay * plasticity[i];
        }
      }
    }
    for (let i = 0; i < size; i += 1) {
      this.weight[i] = Math.min(Math.max(this.weight[i], -config.weightClip), config.weightClip);
    }

    if (bias !== null && biasDirection !== null && isDescent) {
      for (let outIdx = 0; outIdx < outFeatures; outIdx += 1) {
        bias[outIdx] += config.learningRate * biasDirection[outIdx];
        bias[outIdx] = Math.min(Math.max(bias[outIdx], -config.weightClip), config.weightClip);
      }
    }

    let modeFactor = 1;
    if (options.mode === "exploit") {
      modeFactor = 1.2;
    } else if (options.mode === "neighbor-follow") {
      modeFactor = 0.9;
    }
    const reinforcement = new Float64Array(size);
    for (let i = 0; i < size; i += 1) {
      reinforcement[i] = Math.min(cofire[i] + Math.abs(signal[i]), 4.0) * modeFactor;
    }

    const shortEvaporation =
      config.shortPheromoneEvaporation ?? Math.min(0.8, config.evaporation * 4.0);
    const longEvaporation = config.longPheromoneEvaporation ?? config.evaporation;
    const shortReinforcement =
      config.shortPheromoneReinforcement ?? config.pheromoneReinforcement * 2.0;
    const longReinforcement =
      config.longPheromoneReinforcement ?? config.pheromoneReinforcement;

    for (let i = 0; i < size; i += 1) {
      this.shortPheromone[i] *= 1 - shortEvaporation * plasticity[i];
      this.shortPheromone[i] += shortReinforcement * reinforcement[i] * accepted[i];
      this.shortPheromone[i] = Math.min(
        Math.max(this.shortPheromone[i], config.minPheromone),
        config.maxPheromone,
      );
      this.shortPheromone[i] *= connectionMask[i];
    }

    for (let i = 0; i < size; i += 1) {
      this.pheromone[i] *= 1 - longEvaporation * plasticity[i];
      this.pheromone[i] += longReinforcement * reinforcement[i] * accepted[i];
      this.pheromone[i] = Math.min(
        Math.max(this.pheromone[i], config.minPheromone),
        config.maxPheromone,
      );
      this.pheromone[i] *= connectionMask[i];
    }

    this.updateConsolidation(accepted, reinforcement, config, options.consolidate);
    this.lastUpdateMask.set(accepted);

    return acceptedCount;
  }

  private plasticityScale(config: LocalLearningConfig): Float64Array {
    const size = this.neighborIndices.length;
    const scale = new Float64Array(size);
    if (config.consolidationStrength <= 0) {
      scale.fill(1);
      return scale;
    }
    for (let i = 0; i < size; i += 1) {
      const protection = Math.min(Math.max(this.consolidation[i] * config.consolidationStrength, 0), 1);
      const value = Math.max(1 - protection, config.consolidationLrFloor);
      scale[i] = this.connectionMask[i] ? value : 0;
    }
    return scale;
  }

  private updateConsolidation(
    selected: Uint8Array,
    reinforcement: Float64Array,
    config: LocalLearningConfig,
    consolidate: boolean,
  ): void {
    const size = this.neighborIndices.length;
    if (config.consolidationStrength <= 0) {
      this.consolidation.fill(0);
      return;
    }
    for (let i = 0; i < size; i += 1) {
      this.consolidation[i] *= 1 - config.consolidationDecay;
    }
    if (!consolidate) {
      for (let i = 0; i < size; i += 1) {
        this.consolidation[i] *= this.connectionMask[i];
      }
      return;
    }
    const span = Math.max(config.maxPheromone - config.consolidationThreshold, 1e-6);
    for (let i = 0; i < size; i += 1) {
      const mature = Math.min(
        Math.max((this.pheromone[i] - config.consolidationThreshold) / span, 0),
        1,
      );
      const growth =
        config.consolidationGrowth * Math.max(reinforcement[i], 0) * selected[i] * mature;
      this.consolidation[i] = Math.min(Math.max(this.consolidation[i] + growth, 0), 1);
      this.consolidation[i] *= this.connectionMask[i];
    }
  }

  private selectSynapses(
    signal: Float64Array,
    budgetPerOutput: number,
    mode: string,
    followDistance: number,
  ): Uint8Array {
    const { outFeatures, maxNeighbors, connectionMask, pheromone } = this;
    const size = this.neighborIndices.length;
    const selected = new Uint8Array(size);
    const budget = Math.max(1, Math.floor(budgetPerOutput));

    for (let outIdx = 0; outIdx < outFeatures; outIdx += 1) {
      const base = outIdx * maxNeighbors;
      let validCount = 0;
      for (let slot = 0; slot < maxNeighbors; slot += 1) {
        if (connectionMask[base + slot]) {
          validCount += 1;
        }
      }
      if (validCount === 0) {
        continue;
      }
      const k = Math.min(budget, validCount);

      const scores: Array<[number, number]> = [];
      let maxFinite = -Infinity;
      for (let slot = 0; slot < maxNeighbors; slot += 1) {
        if (!connectionMask[base + slot]) {
          continue;
        }
        const score = Math.abs(signal[base + slot]) * (0.5 + pheromone[base + slot]);
        scores.push([slot, score]);
        if (score > maxFinite) {
          maxFinite = score;
        }
      }
      if (maxFinite <= 0) {
        continue;
      }

      if (mode === "neighbor-follow") {
        const frontier = this.recentFrontier(outIdx, followDistance);
        let any = false;
        for (let slot = 0; slot < maxNeighbors; slot += 1) {
          if (frontier[slot]) {
            any = true;
            break;
          }
        }
        if (any) {
          const bonus = Math.max(maxFinite, 1.0);
          for (const item of scores) {
            if (frontier[item[0]]) {
              item[1] += bonus;
            }
          }
        }
      }

      scores.sort((left, right) => right[1] - left[1]);
      for (let t = 0; t < k; t += 1) {
        selected[base + scores[t][0]] = 1;
      }
    }

    for (let i = 0; i < size; i += 1) {
      selected[i] = selected[i] && connectionMask[i] ? 1 : 0;
    }
    return selected;
  }

  private recentFrontier(outIdx: number, followDistance: number): Uint8Array {
    const { maxNeighbors, connectionMask, neighborIndices, inputPositions } = this;
    const base = outIdx * maxNeighbors;
    const frontier = new Uint8Array(maxNeighbors);
    const recentPositions: Matrix = [];
    for (let slot = 0; slot < maxNeighbors; slot += 1) {
      if (this.lastUpdateMask[base + slot] && connectionMask[base + slot]) {
        recentPositions.push(inputPositions[neighborIndices[base + slot]]);
      }
    }
    if (recentPositions.length === 0) {
      for (let slot = 0; slot < maxNeighbors; slot += 1) {
        frontier[slot] = connectionMask[base + slot];
      }
      return frontier;
    }
    for (let slot = 0; slot < maxNeighbors; slot += 1) {
      if (!connectionMask[base + slot]) {
        continue;
      }
      const position = inputPositions[neighborIndices[base + slot]];
      let nearest = Infinity;
      for (const recent of recentPositions) {
        const distance = euclidean(position, recent);
        if (distance < nearest) {
          nearest = distance;
        }
      }
      if (nearest <= followDistance) {
        frontier[slot] = 1;
      }
    }
    return frontier;
  }
}

function activationValue(name: ActivationName, value: number): number {
  if (name === "tanh") {
    return Math.tanh(value);
  }
  if (name === "relu") {
    return value > 0 ? value : 0;
  }
  if (name === "sigmoid") {
    return 1 / (1 + Math.exp(-value));
  }
  return value;
}

function activationDerivativeValue(name: ActivationName, pre: number): number {
  if (name === "tanh") {
    const activated = Math.tanh(pre);
    return 1 - activated * activated;
  }
  if (name === "relu") {
    return pre > 0 ? 1 : 0;
  }
  if (name === "sigmoid") {
    const activated = 1 / (1 + Math.exp(-pre));
    return activated * (1 - activated);
  }
  return 1;
}

export class LocalPheromoneNetwork {
  readonly inputSize: number;
  readonly hiddenSizes: number[];
  readonly outputSize: number;
  readonly activation: ActivationName;
  readonly maxNeighbors: number;
  readonly config: LocalLearningConfig;
  readonly layers: LocalPheromoneLayer[];
  budgetPerOutput: number;
  previousLoss: number | null = null;

  constructor(
    inputSize: number,
    hiddenSizes: number[],
    outputSize: number,
    options: LocalPheromoneNetworkOptions = {},
  ) {
    this.inputSize = Math.trunc(inputSize);
    this.hiddenSizes = hiddenSizes.map((size) => Math.trunc(size));
    this.outputSize = Math.trunc(outputSize);
    this.activation = options.activation ?? "tanh";
    this.maxNeighbors = options.maxNeighbors ?? 8;

    const config: LocalLearningConfig = {
      ...DEFAULT_LOCAL_LEARNING_CONFIG,
      ...(options.config ?? {}),
    };
    if (config.maxBudgetPerOutput === null) {
      config.maxBudgetPerOutput = this.maxNeighbors;
    }
    this.config = config;

    const layerSizes = [this.inputSize, ...this.hiddenSizes, this.outputSize];
    const hiddenShapes = options.hiddenShapes ?? null;
    let hiddenShapeList: (number[] | null)[];
    if (hiddenShapes === null) {
      hiddenShapeList = new Array<number[] | null>(this.hiddenSizes.length).fill(null);
    } else {
      if (hiddenShapes.length !== this.hiddenSizes.length) {
        throw new Error("hiddenShapes must match hiddenSizes");
      }
      hiddenShapeList = hiddenShapes.slice();
    }
    const unitShapes: (number[] | null)[] = [
      options.inputShape ?? null,
      ...hiddenShapeList,
      options.outputShape ?? null,
    ];

    const tagSets: number[][] = [];
    for (let idx = 0; idx < layerSizes.length; idx += 1) {
      const size = layerSizes[idx];
      const shape = unitShapes[idx];
      if (idx === 0) {
        tagSets.push(resolveTags(options.inputTags ?? null, size, options.tagCount ?? 4, shape));
      } else if (idx === layerSizes.length - 1) {
        tagSets.push(resolveTags(options.outputTags ?? null, size, options.tagCount ?? 4, shape));
      } else {
        tagSets.push(defaultTags(size, options.tagCount ?? 4, shape));
      }
    }

    const layers: LocalPheromoneLayer[] = [];
    for (let idx = 0; idx < layerSizes.length - 1; idx += 1) {
      const layerSeed = options.seed == null ? null : options.seed + idx;
      layers.push(
        new LocalPheromoneLayer(layerSizes[idx], layerSizes[idx + 1], {
          maxNeighbors: this.maxNeighbors,
          connectionRadius: options.connectionRadius ?? null,
          tagCount: options.tagCount ?? 4,
          tagDistance: options.tagDistance ?? 0,
          inputShape: unitShapes[idx],
          outputShape: unitShapes[idx + 1],
          inputTags: tagSets[idx],
          outputTags: tagSets[idx + 1],
          shortPheromoneWeight: config.shortPheromoneWeight,
          longPheromoneWeight: config.longPheromoneWeight,
          seed: layerSeed,
        }),
      );
    }
    this.layers = layers;

    const initialBudget =
      config.initialBudgetPerOutput ?? Math.max(1, Math.floor(this.maxNeighbors / 2));
    this.budgetPerOutput = Math.min(
      Math.max(initialBudget, config.minBudgetPerOutput),
      config.maxBudgetPerOutput,
    );
  }

  forwardWithCache(x: Matrix): { output: Matrix; activations: Matrix[]; preActivations: Matrix[] } {
    const activations: Matrix[] = [x];
    const preActivations: Matrix[] = [];
    let h = x;
    for (let idx = 0; idx < this.layers.length; idx += 1) {
      const z = this.layers[idx].forward(h);
      preActivations.push(z);
      if (idx === this.layers.length - 1) {
        h = z;
      } else {
        h = z.map((row) => row.map((value) => activationValue(this.activation, value)));
      }
      activations.push(h);
    }
    return { output: h, activations, preActivations };
  }

  forward(x: Matrix): Matrix {
    return this.forwardWithCache(x).output;
  }

  localTrainStep(x: Matrix, target: Matrix, outputMask?: number[]): TrainStepReport {
    const { output: prediction, activations, preActivations } = this.forwardWithCache(x);

    const rows = prediction.length;
    let loss: number;
    if (outputMask === undefined) {
      let sum = 0;
      let count = 0;
      for (let row = 0; row < rows; row += 1) {
        for (let outIdx = 0; outIdx < this.outputSize; outIdx += 1) {
          const diff = prediction[row][outIdx] - target[row][outIdx];
          sum += diff * diff;
          count += 1;
        }
      }
      loss = count > 0 ? sum / count : 0;
    } else {
      let squared = 0;
      let maskWeight = 0;
      for (let row = 0; row < rows; row += 1) {
        for (let outIdx = 0; outIdx < this.outputSize; outIdx += 1) {
          const weight = outputMask[outIdx] ?? 0;
          const diff = prediction[row][outIdx] - target[row][outIdx];
          squared += diff * diff * weight;
          maskWeight += weight;
        }
      }
      loss = squared / Math.max(maskWeight, 1);
    }

    const previousLoss = this.previousLoss;
    const { mode, improved } = this.adaptBudget(loss);
    const consolidate =
      this.config.consolidationLossGate === null || loss <= this.config.consolidationLossGate;

    const outputError = zeros(rows, this.outputSize);
    for (let row = 0; row < rows; row += 1) {
      for (let outIdx = 0; outIdx < this.outputSize; outIdx += 1) {
        let error = target[row][outIdx] - prediction[row][outIdx];
        if (outputMask !== undefined) {
          error *= outputMask[outIdx] ?? 0;
        }
        outputError[row][outIdx] = error;
      }
    }

    const layerErrors: Matrix[] = new Array<Matrix>(this.layers.length);
    let signal = outputError;
    for (let idx = this.layers.length - 1; idx >= 0; idx -= 1) {
      if (idx < this.layers.length - 1) {
        const pre = preActivations[idx];
        signal = signal.map((row, rowIdx) =>
          row.map((value, outIdx) => value * activationDerivativeValue(this.activation, pre[rowIdx][outIdx])),
        );
      }
      layerErrors[idx] = signal;
      if (idx > 0) {
        signal = this.layers[idx].feedbackToInput(signal);
      }
    }

    let activeSynapses = 0;
    for (let idx = 0; idx < this.layers.length; idx += 1) {
      activeSynapses += this.layers[idx].localUpdate(activations[idx], activations[idx + 1], layerErrors[idx], {
        budgetPerOutput: this.budgetPerOutput,
        mode,
        config: this.config,
        consolidate: consolidate && mode === "exploit",
      });
    }

    return {
      loss,
      previousLoss,
      improved,
      mode,
      budgetPerOutput: this.budgetPerOutput,
      activeSynapses,
      prediction,
    };
  }

  private adaptBudget(loss: number): { mode: string; improved: boolean | null } {
    const previous = this.previousLoss;
    this.previousLoss = loss;

    if (previous === null) {
      return { mode: "warmup", improved: null };
    }
    if (loss < previous - this.config.lossTolerance) {
      const next = Math.floor(this.budgetPerOutput * this.config.shrinkFactor);
      this.budgetPerOutput = Math.max(this.config.minBudgetPerOutput, next);
      return { mode: "exploit", improved: true };
    }
    if (loss > previous + this.config.lossTolerance) {
      const grown = Math.ceil(this.budgetPerOutput * this.config.growFactor);
      this.budgetPerOutput = Math.min(
        this.config.maxBudgetPerOutput ?? this.maxNeighbors,
        Math.max(grown, this.budgetPerOutput + 1),
      );
      return { mode: "neighbor-follow", improved: false };
    }
    return { mode: "steady", improved: false };
  }
}
