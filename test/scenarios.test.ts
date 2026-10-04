import assert from "node:assert/strict";
import test from "node:test";

import { RecallKernel, ngramEmbed } from "../src/index";

const dim = 256;

function rank(candidates: string[], query: string) {
  const kernel = new RecallKernel(dim, { codeDim: 128, seed: 17 });
  const vectors = candidates.map((text) => ngramEmbed(text, dim));
  for (const vector of vectors) {
    kernel.observe(vector);
  }
  const queryVector = ngramEmbed(query, dim);
  return candidates
    .map((text, index) => ({ text, score: kernel.score(queryVector, vectors[index]) }))
    .sort((left, right) => right.score - left.score);
}

const scenarios = [
  {
    name: "Tool cache: near-duplicate request finds the reusable result",
    candidates: [
      "GET /api/orders/42?include=items",
      "GET /api/users/42?include=profile",
      "POST /api/orders/42/refund",
    ],
    query: "GET /api/orders/42?include=item",
    expected: "GET /api/orders/42?include=items",
  },
  {
    name: "Agent memory: a stated preference is recalled for the next turn",
    candidates: [
      "User prefers concise English summaries",
      "User prefers CSV exports for reports",
      "Project deploys to the staging environment",
    ],
    query: "prefers concise English summary",
    expected: "User prefers concise English summaries",
  },
  {
    name: "Support tickets: a similar password-reset issue is surfaced",
    candidates: [
      "Password reset email link expired before use",
      "Invoice email attachment failed to upload",
      "Password policy blocks account creation",
    ],
    query: "Password reset link in email has expired",
    expected: "Password reset email link expired before use",
  },
  {
    name: "Incident logs: a matching gateway timeout is surfaced",
    candidates: [
      "Payment gateway timeout error 504",
      "Inventory database connection error 500",
      "Payment receipt delivery failed 503",
    ],
    query: "Payment gateway timed out with 504 error",
    expected: "Payment gateway timeout error 504",
  },
] as const;

for (const scenario of scenarios) {
  test(scenario.name, (t) => {
    const ranked = rank([...scenario.candidates], scenario.query);
    assert.equal(ranked[0].text, scenario.expected);
    assert.ok(ranked[0].score > ranked[1].score);
    t.diagnostic(`top=${ranked[0].score.toFixed(4)}, runner-up=${ranked[1].score.toFixed(4)}`);
  });
}

test("Memory maintenance: decay lowers pheromone mass after observations", (t) => {
  const kernel = new RecallKernel(dim, { codeDim: 128, seed: 17 });
  for (const text of ["User prefers concise English summaries", "Project deploys to staging"]) {
    kernel.observe(ngramEmbed(text, dim));
  }
  const before = kernel.pheromoneMass();
  const affected = kernel.decayByFactor(0.5);
  const after = kernel.pheromoneMass();
  assert.ok(affected > 0);
  assert.ok(after < before);
  t.diagnostic(`before=${before.toFixed(4)}, after=${after.toFixed(4)}, affected=${affected}`);
});
