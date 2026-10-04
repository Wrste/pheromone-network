/**
 * pheromone_network 公共出口。
 *
 * 只聚合信息素内核三件套，互不冲突：
 * - ./pheromone：零依赖引擎（稀疏局部层 / 双尺度信息素 / 手工 Hebbian）
 * - ./recall   ：召回内核（确定性折叠 + 稀疏自联想投影 + 归一化余弦打分）
 * - ./embed    ：确定性字符 n-gram 编码与点积
 */

export * from "./pheromone";
export * from "./recall";
export * from "./embed";
