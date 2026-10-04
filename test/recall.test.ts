import assert from "node:assert/strict";
import test from "node:test";

import { RecallKernel, cosine, ngramEmbed, normalizedCosine } from "../src/index";

test("内核打分: 归一化余弦不被向量模长带偏", () => {
  assert.equal(normalizedCosine([3, 0, 0], [7, 0, 0]), 1);
  assert.equal(normalizedCosine([2, 2], [3, 3]), 1);
  assert.equal(normalizedCosine([1, 0], [0, 1]), 0);
  assert.ok(Math.abs(cosine([2, 2], [3, 3]) - 12) < 1e-9, "旧 cosine 实为点积");
});

test("RecallKernel: 自联想学习降低重建损失并驱动稀疏连接", () => {
  const kernel = new RecallKernel(128, { tagCount: 16, tagDistance: 1, maxNeighbors: 4, seed: 11 });
  const vectors = ["客户偏好邮件沟通", "发票抬头是Acme科技", "结算周期是月结30天"].map((text) =>
    ngramEmbed(text, 128),
  );

  let firstEpoch = 0;
  for (const vector of vectors) {
    firstEpoch += kernel.observe(vector).loss;
  }
  firstEpoch /= vectors.length;

  for (let epoch = 1; epoch < 30; epoch += 1) {
    for (const vector of vectors) {
      kernel.observe(vector);
    }
  }

  let lastEpoch = 0;
  for (const vector of vectors) {
    lastEpoch += kernel.observe(vector).loss;
  }
  lastEpoch /= vectors.length;

  assert.equal(kernel.trained, true);
  assert.equal(kernel.trainSteps, 31 * vectors.length);
  assert.ok(lastEpoch < firstEpoch, `平均重建损失 ${lastEpoch} 应低于 ${firstEpoch}`);
  assert.ok(kernel.score(vectors[0], vectors[0].slice()) > 0.999);
});

test("RecallKernel: 蒸发与衰减降低信息素质量", () => {
  const kernel = new RecallKernel(128, { tagCount: 16, tagDistance: 1, maxNeighbors: 4, seed: 7 });
  for (const text of ["客户偏好邮件沟通", "发票抬头是Acme科技", "结算周期是月结30天"]) {
    kernel.observe(ngramEmbed(text, 128));
  }
  const before = kernel.pheromoneMass();
  assert.ok(before > 0);
  const affected = kernel.evaporate(0.5);
  const after = kernel.pheromoneMass();
  assert.ok(affected > 0);
  assert.ok(after < before, `信息素质量 ${after} 应低于 ${before}`);
});
