import { createHash } from "node:crypto";

export type Embedder = (text: string) => number[];

/**
 * Deterministic, dependency-free character n-gram encoder.
 * Mirrors the Python featurize(): char 2-5 grams, hashed, L2-normalized.
 */
export function ngramEmbed(text: string, dim = 4096): number[] {
  const normalized = String(text).toLowerCase().trim().split(/\s+/).filter(Boolean).join(" ");
  const vector = new Array<number>(dim).fill(0);
  if (normalized.length === 0) {
    return vector;
  }
  const padded = `^${normalized}$`;
  const grams: string[] = [];
  for (const width of [2, 3, 4, 5]) {
    for (let i = 0; i + width <= padded.length; i += 1) {
      grams.push(padded.slice(i, i + width));
    }
  }
  for (const gram of grams) {
    const digest = createHash("blake2b512").update(gram, "utf8").digest();
    const index = Number(digest.readBigUInt64LE(0) % BigInt(dim));
    vector[index] += 1;
  }
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  const scale = Math.max(norm, 1);
  return vector.map((value) => value / scale);
}

export function cosine(left: number[], right: number[]): number {
  if (left.length !== right.length) {
    return 0;
  }
  let sum = 0;
  for (let i = 0; i < left.length; i += 1) {
    sum += left[i] * right[i];
  }
  return sum;
}
