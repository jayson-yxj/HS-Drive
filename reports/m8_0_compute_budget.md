# M8-0 计算成本、时效与控制周期

**现有 M7-E 不具备在线规划资格。200ms 不放宽，5Hz Planner / 20Hz Controller 不变；单次计算低于 200ms 也不足以证明计划连续有效。** 本文基于 `b3dab19`，不修改任何计时或控制代码。

## 实测计算成本

主证据是 [M7-E 完整交付报告](m7e_full_payload_latency.json)中的 `clean_paired`，以及对应 `m7e_replays/*_clean.json` 原始记录。两个开发序列各 61 个连续快照，无并发诊断；不是相同输入反复预热。`A_uncached` 指未加 M7-E 静态区域缓存，仍含既有几何复用；A 加该缓存；B 是 A 再加 F 候选。

| 序列/臂 | P50 ms | P95 ms | P99 ms | Max ms | 过期/61 |
|---|---:|---:|---:|---:|---:|
| constant A_uncached |391.47|410.45|420.35|430.74|61|
| constant A |391.38|410.49|413.20|414.34|61|
| constant B |624.20|646.14|651.31|652.41|61|
| braking A_uncached |434.49|453.54|457.17|460.65|60|
| braking A |436.62|459.42|462.78|465.13|60|
| braking B |655.86|676.84|690.96|695.51|60|

统计从读取当前快照开始，到完整科学结果 JSON 字节在内存可读为止，包含对象构造和序列化；不含后续报告封装/落盘、真实 IPC 和 Controller。地图准备另耗约 1.38–1.42s，不计入表中稳态耗时，不能在在线首帧忽略。全 campaign 另有并发诊断影响，671 次中 666 次过期；不能混合 clean 与 campaign 分位数。

B 臂每快照组件（constant / braking，单位 ms）：

| 组件 | constant P50 / P95 | braking P50 / P95 | 计时含义 |
|---|---:|---:|---|
| 快照解码/校验 |0.052 / 0.063|0.049 / 0.080|不含真实状态采集和 IPC |
| Snapshot → Request |0.086 / 0.093|0.085 / 0.096|对象转换/输入合同 |
| 路线与 Actor 道路查询 |4.903 / 5.096|4.346 / 8.350|当前路线支持，不是全部 exact 查询 |
| 运动假设 |2.538 / 2.585|2.525 / 2.575|有限假设族构造 |
| Actor Polygon 准备 |37.545 / 38.302|37.769 / 38.551|本快照各假设车身复用 |
| 假设未来全车身道路支持 |20.091 / 20.467|24.576 / 25.286|不能只检查中心线 |
| 原候选生成 |4.445 / 4.943|2.944 / 3.358|含原慢行生成 |
| 新 F 生成 |1.720 / 1.870|1.704 / 1.907|完整审核在下表 |
| 选择 |0.009 / 0.014|0.007 / 0.013|不是规划总成本 |
| 返回对象构造 |45.033 / 52.237|41.288 / 42.508|逐候选及请求/假设 asdict 等 |
| JSON 序列化 |8.640 / 9.048|8.146 / 8.839|尚未含进程交付 |

每候选 P95（道路/车身，动态 SAT，停车余量），单位 ms：

| 候选 | constant | braking | Guide 点数中位数 constant / braking |
|---|---|---|---:|
| A_keep_route |31.76 / 12.72 / 不适用|80.49 / 12.78 / 不适用|161 / 161|
| B_left |93.80 / 12.64 / 不适用|无该候选样本|321 / —|
| D_stop |30.50 / 48.81 / 不适用|79.78 / 49.01 / 不适用|161 / 161|
| E_slow_follow |30.42 / 12.60 / 15.49|81.15 / 12.88 / 15.75|161 / 161|
| F_state_anchored_slow |197.86 / 12.66 / 15.25|198.42 / 12.84 / 15.74|716 / 727.5|

F 的 braking 样本为 40 条，其他该表候选通常为 61 条；未成功构造的候选不能计作零成本样本。道路/车身计时包含区域准备、完整 guide、sweep、exact Waypoint 和其他几何约束，现有 clean 数据**没有独立的 guide/exact 子计时**。表中 P95 不能相加得到完整 P95，返回对象等部分也与旧 `timing_ms.total` 有重叠。总交付以 `delivery.full_ms` 为准。

辅助 [M7-D cProfile](m7d_execution_logs/function_profile.txt) 只覆盖三个开发快照且有 profiler 开销：6658 次 Waypoint 累计约 0.309s；12 次动态评价约 0.255s，其中 156 次向量 SAT 约 0.042s；105 次 asdict 约 0.364s；3 次 Polygon 准备约 0.175s；3 次停车余量约 0.060s；候选生成约 0.021s。union/buffer 约 0.044/0.025s。累计时间相互嵌套，不能直接加总或外推线上 WCET。

因此：只优化 JSON、只编译 SAT、只缓存 union/buffer 都缺乏单独解决约 650ms 完整耗时的依据；完整道路/车身审核及对象展开是必须联合审查的成本。

## 缓存的真实收益与边界

静态 region cache：A 两序列命中 242/244、182/183，B 为 303/305、222/223，约 99%。A 的完整 P95 仍为 410.49/459.42ms，对照 410.45/453.54ms，没有可重复的实质提速。

混合 body/exact/prepared cache 的累计命中与未命中：constant A 为 84615/147675，B 为 90483/273288；braking A 为 80380/207226，B 为 82780/296296。由此算出的约 36.43%/24.87%、27.95%/21.84% 包含同帧与不同类型查询，**不是纯跨帧命中率**。

源码归因：prepared candidate 身份含完整轨迹及 stamp，移动快照不能直接复用；exact 点身份含原始 XY/Z 与允许车道，位姿/Z 改变造成失效；prepare 在部分廉价拒绝之前构建全轨迹和 guide sweep，stop 亦有后续未使用的 guide 准备；region 每次构造 WKB 键，字典没有显式容量上界。当前不缓存动态通过结论是应保留的性质。参考：[缓存代码](../diagnostics/m7c_geometry.py)、[region](../diagnostics/m7e_region.py)。

## 两种时间合同必须明确区分

| 现存合同 | 原始依据 | 边界 |
|---|---|---|
| M7-D/E 研究 | [M7-E protocol](../configs/m7e/protocol.json)、[时序合同](../datasets/m7d_temporal.py) | 原 snapshot 时间 + **0.2s**；完整交付预算 **200ms** |
| 历史在线 Worker | [m5a_async](../simulator/m5a_async.py)、[m6b_worker](../simulator/m6b_worker.py)、[m6c_worker](../simulator/m6c_worker.py) | 默认 `max_age=0.35`，检查仿真年龄与原输入墙钟年龄；该行为本轮不变 |

0.35s 是既有实现事实，不是本次允许 M7 延长到 350ms 的理由；M7 的 0.2s 也不能被写成历史所有 Worker 已使用的默认值。下一代准入需显式遵守不宽于 M7 的 200ms 边界，另外版本化适配，不能偷偷改旧 Worker 默认值。

M7-E 回放的 `modeled_sim_available = original_sim + full_wall_duration` 是非实时回放估计。采集时的 monotonic 墙钟与后来回放墙钟不能相减解释为 live snapshot 年龄；当前表不具备真实 IPC、队列或仿真推进年龄证据。

下一代必须分别记录源 snapshot 的 sim/frame/available-wall、入队、出队、开始/完成计算、payload 就绪、收到结果、Safety 开始/结束、控制命令、CARLA tick 返回。需同时报告输入年龄、排队、核心计算、对象/编码、IPC、结果年龄、Safety、控制及 tick；不得用完成时刻重新盖源时间戳。

## 5Hz 与 200ms 有效期的交接约束

若快照采样时刻 `t_k = k × 200ms`，有效期到 `t_k + 200ms`，本次延迟为 `L_k > 0`，下一结果最早在 `t_k + 200ms + L_(k+1)` 就绪，那么旧结果到期与新结果就绪之间存在 `L_(k+1)` 的空档。即便每个 `L_k < 200ms`，也不能证明连续有新鲜规划。

例如 L=80ms：源时间 0 的计划在 80ms 就绪、200ms 到期；源时间 200ms 的计划在 280ms 就绪。20Hz 的 250ms 控制时刻没有有效计划。200ms 恰好边界的比较符号和时钟抖动不能消除这个问题。

这需要未来单独证明调度相位及每个控制时刻的有效覆盖，或声明并验证空档中的降级行为。不能以六秒轨迹长度代替 200ms 状态有效期，也不能通过重新盖时间戳、延长预算、未经审核提高频率或只检测平均 Hz 来解决。若提出最新状态重新认证，必须是真实执行新鲜状态下必需安全检查的独立合同，不是为旧计划“续期”；本轮没有批准该设计。

## 下一代预算提案，不是通过标准的实测值

规划请求从源状态到可用控制的**所有项**共用 200ms。为避免核心占满周期，可先按以下研究预算审查：输入已有年龄 20ms、dispatch/排队 10ms、核心 100ms、编码/IPC/消费 10ms、Safety 15ms、Controller/命令提交 5ms、余量 40ms，共 200ms。每项是设计分配，不是测得能力；单项超支或端到端超时都必须拒绝，剩余时间不能假定总能借到。

核心需有输入规模上界、内存上界、地图准备边界、迭代次数和超时点。候选最多 6、每 Actor 最多 16 个名义假设、六秒 120 个动态区间可作为原协议参考；Actor 上限、guide 复杂度、地图块大小仍待容量基准确定，不能凭空认定 32 Actor 一定能过。超容量必须返回明确不支持，不能静默截掉危险对象。安全相关检查未完成就不能发布部分“通过”。

100ms 是未来原型的核心目标；200ms 仍为完整交付及准入边界。Linux/Python/CARLA 下有限样本 Max 不构成严格 WCET 证明，编译代码也不能自动提供主机调度上界。后续须报告约束运行域内的实测 Max、P99、deadline miss、未覆盖负载及 watchdog 行为，并单独解决上述交接空档。

20Hz 控制周期是 50ms，不能用 worker 200ms 预算替代。控制关键路径中的状态快照、有效期核查、Safety、Controller、apply_control 和 CARLA tick 合计需满足该周期；日志写入及完整研究 JSON 不应阻塞。必要安全检查不能挪到控制以后。无有效计划、坏输入、Worker 崩溃/迟到、时间跳变都进入既有故障停车/紧急保护路径，不使用过期计划正常控制。历史低速停车物理能力仍未验证，故障指令可发出不等于任何情境都能安全停住。

## 真实在线长尾仍独立开放

以下是 [M7-A 汇总的历史 M6-B 实测](m7a_runtime_status.json)，不是 M7-E 在线成绩：

| 项目 | 样本数 | P50 ms | P95 ms | P99 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| 控制间隔 |43423|49.992|50.512|51.171|178.083|
| CARLA world.tick |43492|1.798|2.365|3.201|94.276|
| Worker |10366|39.431|87.259|114.898|569.282|

原统计另有 15 次 deadline miss、16 个 skipped slots、0 次过期计划用于正常控制。deadline miss 的原定义是工作结束越过墙钟槽，不等同于每个控制间隔 >50ms；三种计数要分开。不能将历史较轻 Planner 的耗时当 M7 全检查性能。共享 GPU 的完整六摄模型 + CARLA + 新核心长尾仍 NotRun。RuntimePartial、严格 20Hz 未通过状态不变。

## 复核命令与源码身份

只读汇总命令（不重新运行回放，不覆盖报告）：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('reports/m7e_full_payload_latency.json').read_text())
for group in d['clean_paired']:
    for arm, result in group['modes'].items():
        print(group['episode'], arm, result['full_ms'], 'expired', result['expired'])
print('campaign snapshots', sum(x['snapshots'] for x in d['episodes']))
print('campaign expired', sum(x['expired'] for x in d['episodes']))
f = json.loads(Path('reports/m7e_actual_future_collision_audit.json').read_text())
print('accepted/conflicts', sum(x['accepted_total'] for x in f['episodes']),
      sum(x['accepted_conflict_candidates'] for x in f['episodes']))
PY
```

本轮核验 SHA256：`tools/m7e_replay.py` = `a1b7af51e464e0eec97614dd38d6820d2715633e221d327e78b8a5cd92c251f5`；`diagnostics/m7d_geometry.py` = `ace04d6d6fcbb44142dd8c85e129953d17a22eea26fc002ad40c158d2fac3254`；`diagnostics/m7e_region.py` = `9c6b8a6ce9ab573894621fe9cdedbdc426d599f0ffc67af5996ba17f62402d7c`；`simulator/m5a_async.py` = `76f74414b51ef833367a8c6a3bbe21bfe888807badd4742845b160ff00bd0c27`。
