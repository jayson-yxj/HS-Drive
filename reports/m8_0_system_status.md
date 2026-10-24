# M8-0 系统状态与 M7 Research Closeout

**决策：下一主阶段优先恢复范围受控的六摄三维视觉可行性研究；有界在线规划核心列为次优先级。本轮只完成评审，不批准任何模型或在线实现。**

日期：2026-10-24。依据提交：`b3dab191485dfe6da883d0d361064f02522b1a9c`。评审分支：`feat/m8-0-architecture-review`。本文件兼作 M7-A 至 M7-E 的正式结题记录，状态为：

`ResearchCompleted / OfflinePlannerResearchGo / OnlinePlannerNoGo / SafetyNotValidated`

这里的 ResearchCompleted 指预定研究得到可复核的结论并停止自动扩展；OfflinePlannerResearchGo 指可以复用离线评价能力。二者都不授权接入在线 Shadow 或车辆控制。

## 研究成果及证据边界

| 阶段 | 可以封存复用的成果 | 不能据此声称的能力 | 原始记录 |
|---|---|---|---|
| M7-A | Oracle 场景基准、候选/道路/车身检查、六摄 Shadow 数据来源合同；发现恒速假设下的危险接受 | 未知未来安全、六摄空间定位合格 | [总结](m7a_summary.md) |
| M7-B | 按真实道路弧长修复场景，有限加减速/延迟多假设评价；保留历史七条危险候选检出 | 任意横穿意图、所有中间运动参数均安全 | [总结](m7b_summary.md) |
| M7-C | 几何复用、唯一合法后继道路延拓、E 慢行候选；205 场景/1662 候选等价回归 | 相同输入预热耗时代表移动 Ego；路线中心存在代表全车身支持 | [总结](m7c_summary.md) |
| M7-D | 真实 CARLA 时序快照、与规划隔离的实测未来评价、停止/前进覆盖审计 | 反事实 Ego 轨迹已经执行、Actor 会响应该轨迹 | [总结](m7d_summary.md) |
| M7-E | 实测 XY/航向/前向速度锚定的 F 慢行连接，完整结果交付计时，静态区域缓存等价审计 | 真实初始加速度/曲率已测得、实际控制跟踪可行、200ms 在线通过 | [总结](m7e_summary.md) |

M7-E 的最终证据按其原分母保留：

- 历史 **205 个有效场景、1662 条候选**全部等价；历史危险候选 **7/7** 仍被检出。场景含共享模板，候选不是独立驾驶试验。[等价报告](m7e_geometry_equivalence.json)、[验收](m7e_acceptance.json)。
- F 成功构造 581 条，t=0 状态最大残差约 `2.78e-17`；负实测前向速度拒绝生成。计划初始加速度及曲率为零是显式假设，并无可信实测加速度/Yaw Rate 连续性证据。[轨迹报告](m7e_state_anchored_trajectory.json)。
- 新最终 road65/road67 的继续行驶选择由 0/0 变为 27/28，共 **55 次**。这是两段相关时序中的选择增益，不是 55 个独立成功场景。第三段 road28 横穿仍 61/61 拒绝，Person 运动走廊仍 Unknown。[配对结果](m7e_candidate_ablation.json)。
- 8 段既有序列加 3 段新序列，共 **671 个规划时刻**；**638 条接受候选**（其中 F 为 250 条）未检测出实测未来 0.05s 采样 OBB 冲突。接受候选分母不同于规划时刻分母。此参考是有限采样平面车身盒，既非连续 Mesh 真值，也非车辆执行安全证明。[未来审计](m7e_actual_future_collision_audit.json)。
- 完整科学结果在内存中序列化可读时，**666/671 次过期**。真实 IPC、车辆跟踪和共享 GPU 调度未计入该通过证明；它们不能补救已经失败的离线时效。[完整时延](m7e_full_payload_latency.json)。
- 新采集三段无碰撞回调，只证明采集脚本的这几段运行；采集车辆未执行 M7-E Planner 输出。新最终道路距开发实测 Ego 路径最近约 97.10m，仍为同一 Town03，部分道路曾用于合成回归。[场景支持](m7e_fresh_episode_support.json)。

因此 `RobustWithinHypotheses`、`SafeForRecordedFuture` 与 `FeasibleToExecute` 必须继续分开。道路分支意图、横穿运动支持、初始动态状态、停车执行、回滑/保持、有限假设外行为及真实动态交互均未全部验证。

## 整个系统的独立状态

| 能力 | M8-0 继承状态 | 下一决策需要的证据 |
|---|---|---|
| 六摄 RGB/K/T、同步与监督隔离 | 工程合同可复用 | 适配新框架的实测 I/O、坐标及访问白名单 |
| HeightSplat / 四层 BEV / Spatial Evidence V2 | 接口可复用，质量仍有限 | 可见障碍空间定位及泛化，Free-Ray 不等于整格可通行 |
| M5 冻结 Mobile + DirectCenterHead | `ResearchCompleted / CandidateNoGo / SafetyNotValidated` | 不继续扩训；成熟方法的独立可行性证据 |
| M7 离线 Oracle 规划 | `OfflinePlannerResearchGo` | 保留反例、有效场景和完整评价器 |
| M7 在线 Planner / 在线 Shadow | `OnlinePlannerNoGo / NotApproved` | 有界计算、端到端时效、连续交接与新鲜状态安全检查 |
| Oracle 真实车辆执行 | 原 `ControlNoGo` 等未闭合项不变 | 动力学、停车/保持、跟踪、任务完成与故障回归 |
| 5Hz/20Hz Runtime | `RuntimePartial`，严格 20Hz 未通过 | 控制周期长尾、共享资源、故障与最坏输入负载 |
| 可靠 3D 感知 / 正式 Occupancy | 未通过 | 独立新位置、全候选、风险覆盖与空间监督质量 |
| Temporal Memory / Sparse Query 实际质量 | 未验证 | 单帧可信基线后单独立项 |
| RGB-only 驾驶安全 / Bench2Drive 官方成绩 | `SafetyNotValidated / NotRun` | 独立授权及合格视觉 Agent；当前没有官方分数 |

长期架构仍遵循[技术基线](../docs/HS-Drive_Technical_Plan_v1.0.md)的 Dense BEV、Sparse Query、Occupancy、Temporal Memory 与规划模块组合。M4.8 后非互斥空间证据合同及后续实测安全限制优先于早期方案中的未验证设想。本次改变研发顺序，不重新设计整个网络，不将规则 Oracle Planner 视为最终视觉学习规划器。

数据来源仍由 [M6-A 接口](../datasets/m6a_interfaces.py)及 [M7-A 合同](../datasets/m7a_contracts.py)显式标识：Oracle 为 `Privileged`；RGBSpatialEvidence 和 DetectedObjectHypothesis 仅 Offline Shadow；Unknown 不是安全 Free。保存原始帧号、仿真时间、墙钟、坐标系、版本和有效期。现有类型检查是工程边界，并非权限认证机制，仍需访问隔离及接线审查。

## 封存与本轮实际检查

本轮实际执行的是源码/报告读取、JSON 数值复核、资产 SHA256 核验及官方资料查询。没有运行 CARLA、GPU 前向或训练。M7-E 原测试结果为 **466 passed / 8 skipped / 1 xfailed，CARLA SDK 专项 67 passed**；这是历史测试，不是本轮重新运行。

本轮重新核验 [M7-E release manifest](m7e_release_manifest.json) 中 **525/525 文件**的大小和 SHA256，零差异；重新核验六摄 Shadow **40 packets / 240 RGB / 38 位置组**，图像 SHA 零差异。Mobile 和 Direct 代表 Checkpoint SHA 也与历史一致，见[感知盘点](m8_0_perception_readiness.md)。这不等于本轮重新核验全部原始训练库或做了异地备份。

五份交付文档的 57 个本地链接均存在；两段下列/预算文档中的只读 Python 命令实际执行通过。JSON 汇总重新确认 671/666、638/0、205/1662/55；逐候选与组件分位数从 clean 原记录重新计算。只做文档变更，因此未重新运行历史模型测试或 CARLA 测试，也未添加实现镜像测试。

M7-E 历史保全记录核验过 15020 个当时基础跟踪文件、1170 个 M7-D manifest 条目、50867 个继承外部条目及 4 个 CARLA 0.9.16 二进制条目，原记录继续保留；本轮没有重复声称运行该全量检查。[原保全记录](m7e_execution_logs/preservation.json)。失败日志、原始时序、模型、协议和各版本结果原位保留，不把报告汇总替代原始证据。

本轮固定的证据入口 SHA256：

| 文件 | SHA256 |
|---|---|
| `reports/m7e_release_manifest.json` | `ea6f946ae54d651d59c4334c6156abc58fc22d2902bfc25b1f098504c5fd4f89` |
| `reports/m7e_full_payload_latency.json` | `5d5810da80d49d5e10527eaeaeb7797773faa3e8a524f40ee3ab14a4b77a2c99` |
| `reports/m7e_geometry_equivalence.json` | `464f6731ad61b6f819d14bab5a20ef6463e04fe495d4aaba5e418825e42a8424` |
| `reports/m7e_actual_future_collision_audit.json` | `d78b04b3a1b82bd7cb37d96051df9932a356201a5b4a13fe6e44642e7dd76c58` |
| `reports/m5o_artifact_manifest.json` | `104327fdcf9877e7c0e3966fb2c98dd12321e84ef9584e83fa19bd080cb74964` |
| `reports/m7a_release_manifest.json` | `ea8840247fd3c815c905c9cd4d730074881cf1e91d810736195344c2cae1d7dc` |

以下只读命令可重做本轮两项资产检查，不触发实验：

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
git show --no-patch --format=fuller b3dab19
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()
manifest = json.loads(Path('reports/m7e_release_manifest.json').read_text())
for item in manifest['files']:
    p = Path(item['path'])
    assert p.stat().st_size == item['bytes'], p
    assert sha(p) == item['sha256'], p
packets = list(Path('reports/m7a_shadow_packets').glob('*.json'))
images = 0
groups = set()
for path in packets:
    packet = json.loads(path.read_text())
    groups.add(packet['sample_id'].split('/')[0])
    for camera in packet['cameras']:
        assert sha(camera['path']) == camera['sha256'], camera['path']
        images += 1
print('M7E files:', len(manifest['files']), 'mismatches: 0')
print('Shadow packets/images/groups:', len(packets), images, len(groups))
PY
```

决策与理由见 [ADR](../docs/M8_0_ARCHITECTURE_DECISION.md)，计算明细见[预算](m8_0_compute_budget.md)，两种方案见[架构选项](m8_0_architecture_options.md)，视觉与 Bench2Drive 接入前置条件见[就绪度](m8_0_perception_readiness.md)。任何下一研发里程碑均需单独批准。
