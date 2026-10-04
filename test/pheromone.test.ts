import assert from "node:assert/strict";
import test from "node:test";

import { LocalPheromoneNetwork, type LocalLearningConfig, type Matrix } from "../src/index";

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianSampler(seed: number): () => number {
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

function makeRegression(rows: number, seed: number): { x: Matrix; target: Matrix } {
  const gaussian = gaussianSampler(seed);
  const x: Matrix = [];
  const target: Matrix = [];
  for (let row = 0; row < rows; row += 1) {
    const values: number[] = [];
    for (let col = 0; col < 12; col += 1) {
      values.push(gaussian());
    }
    x.push(values);
    target.push([
      0.5 * values[0] - 0.25 * values[1] + 0.2 * values[2],
      -0.3 * values[4] + 0.4 * values[5] + 0.35 * values[7],
      0.15 * values[8] - 0.45 * values[9] + 0.3 * values[11],
    ]);
  }
  return { x, target };
}

function mse(model: LocalPheromoneNetwork, x: Matrix, target: Matrix): number {
  const prediction = model.forward(x);
  let sum = 0;
  let count = 0;
  for (let row = 0; row < prediction.length; row += 1) {
    for (let col = 0; col < prediction[row].length; col += 1) {
      const diff = prediction[row][col] - target[row][col];
      sum += diff * diff;
      count += 1;
    }
  }
  return sum / count;
}

function buildModel(seed: number, overrides: Partial<LocalLearningConfig> = {}): LocalPheromoneNetwork {
  return new LocalPheromoneNetwork(12, [], 3, {
    inputShape: [3, 4],
    outputShape: [3, 1],
    inputTags: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    outputTags: [0, 1, 2],
    maxNeighbors: 4,
    tagCount: 3,
    tagDistance: 0,
    config: {
      learningRate: 0.08,
      initialBudgetPerOutput: 4,
      maxBudgetPerOutput: 4,
      minBudgetPerOutput: 1,
      evaporation: 0.01,
      consolidationStrength: 0.5,
      consolidationGrowth: 0.05,
      consolidationThreshold: 1.05,
      ...overrides,
    },
    seed,
  });
}

test("局部学习收敛: 纯 TS 信息素层让回归 loss 显著下降", () => {
  const { x, target } = makeRegression(128, 7);
  const model = buildModel(3);

  const before = mse(model, x, target);
  for (let step = 0; step < 80; step += 1) {
    model.localTrainStep(x, target);
  }
  const after = mse(model, x, target);

  assert.ok(after < before, `loss 应下降: ${before} -> ${after}`);
  assert.ok(after < before * 0.3, `loss 应显著下降: ${before} -> ${after}`);
});

test("标签分区: tagDistance=0 时每个输出只连接同标签输入", () => {
  const model = buildModel(5);
  const layer = model.layers[0];

  assert.deepEqual(layer.connectedInputs(0).sort((a, b) => a - b), [0, 1, 2, 3]);
  assert.deepEqual(layer.connectedInputs(1).sort((a, b) => a - b), [4, 5, 6, 7]);
  assert.deepEqual(layer.connectedInputs(2).sort((a, b) => a - b), [8, 9, 10, 11]);
});

test("双时间尺度信息素: 被选中的突触被强化并开始固化", () => {
  const { x, target } = makeRegression(128, 7);
  const model = buildModel(9);
  const layer = model.layers[0];

  const initialPheromone = layer.pheromone[0];
  assert.equal(initialPheromone, 1);

  for (let step = 0; step < 120; step += 1) {
    model.localTrainStep(x, target);
  }

  let maxLong = 0;
  let maxShort = 0;
  let maxConsolidation = 0;
  for (let i = 0; i < layer.pheromone.length; i += 1) {
    if (!layer.connectionMask[i]) {
      continue;
    }
    maxLong = Math.max(maxLong, layer.pheromone[i]);
    maxShort = Math.max(maxShort, layer.shortPheromone[i]);
    maxConsolidation = Math.max(maxConsolidation, layer.consolidation[i]);
  }

  assert.ok(maxLong > 1, `长期信息素应被强化: ${maxLong}`);
  assert.ok(maxLong <= 8, `长期信息素应被上限约束: ${maxLong}`);
  assert.ok(maxShort > 0, `短期信息素应保持为正: ${maxShort}`);
  assert.ok(maxLong > maxShort, `长期(跨会话)轨迹应强于短期轨迹: ${maxLong} vs ${maxShort}`);
  assert.ok(maxConsolidation > 0, `成熟突触应开始固化: ${maxConsolidation}`);
});

test("预算自适应: 首步 warmup，后续 mode/预算在约束范围内", () => {
  const { x, target } = makeRegression(64, 11);
  const model = buildModel(13);

  const first = model.localTrainStep(x, target);
  assert.equal(first.mode, "warmup");
  assert.equal(first.previousLoss, null);

  const validModes = new Set(["warmup", "exploit", "neighbor-follow", "steady"]);
  for (let step = 0; step < 40; step += 1) {
    const report = model.localTrainStep(x, target);
    assert.ok(validModes.has(report.mode), `非法 mode: ${report.mode}`);
    assert.ok(report.budgetPerOutput >= 1 && report.budgetPerOutput <= 4, `预算越界: ${report.budgetPerOutput}`);
    assert.ok(report.activeSynapses >= 0);
  }
});
