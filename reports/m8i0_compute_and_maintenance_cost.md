# 计算、维护与复现成本

2026-10-26，以下工作量为工程估计而非测量；没有新GPU或CARLA负载实验。

| 项目 | A 原生 TM | B ScenarioRunner0.9.15 | C Bench2Drive文件 |
|---|---|---|---|
| 新依赖 | 当前CARLA/Python/Numpy、已有采集库 | 旧py-trees/Numpy/Shapely等，需隔离 | 单帧只需现有图像/json/gzip；LiDAR另需laszip，不必先读 |
| CARLA/地图 | 现有0.9.15和一个已安装Town | 0.9.15，按XML核验Map/包SHA | 离线读文件无需服务器/地图；复演需0.9.15附加地图 |
| Tick/进程 | 一个Session owner，TM与Walker后台；被动Evaluator | runner/manager/provider/agent均可能tick；集成更复杂 | 无tick，原时间字段缺失需先处理 |
| 故障与清理 | 自有actor/controller/sensors清单，有限超时、settings恢复 | 行为树中增删/隐藏/恢复对象，复杂度更高 | 包SHA、受限解压、损坏帧/字段拒绝，不执行附带脚本 |
| 放置器维护 | 复用原生流程，无搜索/射线规则扩张 | 已有场景仍含手工offset/替换/随机重试 | 无放置代码，仍需审核源标注 |
| 可复现 | seed+实际spawn/navigation+轨迹+地图/配置hash | 另加场景XML、行为事件与随机状态 | immutable包/文件SHA最强，采集过程透明度较低 |
| 首轮磁盘 | 提案只留最多12组24路数据，设置1GiB上限 | 同等采集+场景log | 候选单包267.2MiB；解压量未知，先设2GiB上限；全Mini2.62GiB压缩 |
| 4090预算 | 仿真渲染，无模型并发；峰值待Pilot | 同类渲染另加场景CPU；未知 | 解码审查CPU即可，GPU0 |
| 初步工程估计 | 2–4开发日：有限owner+Sidecar+审核，不含失败修补 | 4–7日：环境/生命周期/传感器集成，不承诺通过 | 1–2日：单帧合同，许可/网络等待不可估算 |

若提案采用960×576、四模态六摄、每路BGRA4byte，单组未压缩约50.6MiB；12组约607.5MiB，不包含临时队列/解码和日志。1GiB是提案磁盘上限，不是实测压缩比；全时段24路原始保存会远超这一预算。传感器短时回调队列必须有界、丢帧显式失败，不为保磁盘而改帧时间。

复用范围：`simulator/synchronizer.py`的同帧Join/超时、`datasets/m8e_reference.py`等对象来源审查、历史足迹清单及投影函数。`datasets/m8b_petr_adapter.py`强制旧shape/顺序/角度，不可直接喂B2D；正式适配属于后续批准范围。本轮不改变任何默认模块。

准时性不以本轮文本审查推断：M8-B端到端P95约337.94ms、历史共享GPU运行时Partial保持。TM数据来源Pilot即使离线慢跑成功，也不是20Hz驾驶或PETR实时性验收。
