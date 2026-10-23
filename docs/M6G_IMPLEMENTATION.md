# M6-G 实现、字段合同与复现

本轮是M6-F真实数据的离线研究、官方源码审计及实际服务器构建条件检查。**没有启动CARLA、没有新内部量、没有编译或训练。** 原始30次试验全部保留，不将本轮pytest或离线重算写成新增CARLA验证。

## 代码与资产

| 路径 | 职责 |
|---|---|
| `configs/m6g/protocol.json` | 分析前冻结阈值、Applied对齐、停止规则 |
| `diagnostics/m6g_events.py` | 命令转换匹配、连续帧检查、重建差分屏蔽、制动事件度量 |
| `tools/m6g_analyze.py` | 26个历史trace的完整离线事件分析，拒绝覆盖已存在输出 |
| `tools/m6g_details.py` | 四轮首次大滑移前后字段、左截断、Sham首Tick配对 |
| `tools/m6g_figures.py` | Applied制动链、起步链、Sham初态、冲击证据范围图 |
| `tools/m6g_sources.py` | 有限官方文件检索；本轮已固定源码快照，复现无需重新联网 |
| `tools/m6g_reproduce.py` | 新建隔离输出目录，重算并比较19个核心结果字段 |
| `tools/m6g_finalize.py` | 汇总已执行证据、构建失败门禁、NotRun条件包和三项独立决策 |
| `tools/m6g_release.py` | 只读核验历史SHA、封存本阶段清单 |
| `tests/test_m6g_events.py` | 合成延迟、截断、非连续帧、错帧及差分合同 |
| `tests/test_m6g_regression.py` | 真实历史晚帧、Sham缓存、零debug力强减速及14,810帧回归 |

全部新增仅在M6-G路径；正式`simulator`、`planner`、`safety`和模型代码没有改动。没有异步遥测消费者，未声称实现有界队列/丢弃机制。未来若单独开展异步方案，仍必须保留唯一Tick Owner、源帧与时间、队列边界、超时丢弃、错帧拒绝和缺帧统计。

## 原始数据与来源

- `reports/m6f_trials_0916/*/*/trace.jsonl`：26次真实CARLA0.9.16，共14,810外层帧。
- `reports/m6f_guard_0915`：4次真实CARLA0.9.15冻结G1对照，本轮只复核。
- `reports/m6f_execution_logs/trial_manifest.json`：原始采集模块字节、代码对象哈希、提交及文件SHA。
- `reports/m6f_execution_logs/server_binary_manifest.json`：0916安装二进制SHA；完整源构建身份仍NotVerified。
- `reports/m6g_execution_logs/trial_manifest.json`：逐试验trace、时间线、配置、Physics SHA与采集源版本。
- `reports/m6g_event_timelines`：26份完整逐帧派生时间线，原始字段没有平滑。

原始采集提交为可行性`c1e2498`、响应`58a8d05`、Sham/开销`29670eb`、Guard`02e7955`；分析提交`6e9d7d4`、细节与图`30b5988`。每份最终JSON另记录汇总源码SHA。最终提交不是所有采集的执行版本。

主阶段来源标记`Privileged`。14,810帧、轮端记录、重复编号均不当作独立场景样本；没有基于重复帧计算独立因果置信区间。

## 时间、单位与事件合同

1. Request为`apply`墙钟单调时间及请求值；RPC返回时间独立保存。它不等于服务器执行时刻。
2. Applied来自首次匹配的ActorSnapshot，Telemetry回读LastAppliedControl作一致性核验。它仍不是平滑后的最终模拟输入。
3. 命令按throttle/brake/steer/handbrake/reverse变化分段，在下一个请求前未匹配则Unresolved。manual gear请求完整保存，实际Gear变化单独成为事件。
4. 每个Applied事件给出前后模拟快照区间、首次观测frame和读取完成的墙钟上界。真实求解器时间`NotAvailable`；25个延迟事件全部保留。
5. 使用相邻真实timestamp差计算有符号车身速度的原始差分加速度与Jerk，固定外层约50ms。API加速度另列。检查试验内相邻frame和时间，禁止跨缺帧相减。
6. Physics重建后的首个差分及相邻Jerk标为null/invalid，防止旧缓存与重建后live状态混算；世界速度/位置原值仍保存。
7. omega为rad/s；RPM为转/分；long_slip无量纲；API lat_slip为角度；load和debug force为N，debug torque为N·m。原始非有限归一化字段保留M6-F `invalid_raw`约定，缺测不填0。
8. 每轮debug torque不是实际Brake Torque；调试力不是50ms合力积分。API不含独立求解器frame/time，外层bracket无法补出这些字段。

固定描述阈值见协议。首轮下降与车身强减速的“谁先”依赖该描述定义，不称精确运动学起点；大滑移制动前已经存在的标为左截断。冲击候选标签可重叠。证据级别Observed、Correlated、MechanismSupported和Unresolved的适用边界见总结及源码报告。

41个“同帧调试力不足以解释”Jerk峰值中，40个有遥测且接近零，1个ABBA初始化计划关闭遥测，单独列出；缺测不是零力。37个普通小制动的最大减速度点中17个四轮近零，是不同统计。

## 实际执行记录与复现命令

环境沿用现有`.venv-gpu`及`.venv`，未安装新依赖、未改变GPU栈。离线使用numpy/matplotlib，pytest作工程回归；不需要启动CARLA。环境版本另见`m6g_execution_logs/python_environment.json`。

在仓库根目录执行下列命令，自动创建临时输出目录，并与封存事件逐项精确比较；原报告和数据不会被覆盖：

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
.venv-gpu/bin/python -m tools.m6g_reproduce
```

也可指定尚不存在的新目录：

```bash
.venv-gpu/bin/python -m tools.m6g_reproduce --output /tmp/hsdrive-m6g-my-reproduction
```

实际执行顺序是分析→细节→图，每个模块stdout/stderr都落到新目录。复现比较核心事件、排序、帧、数值与Runtime统计；不比较带执行版本的元数据，图片字体/后端元数据也不作字节等价认证。

测试复现：

```bash
.venv-gpu/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_m6g_events.py tests/test_m6g_regression.py tests/test_m6f_telemetry.py tests/test_m6f_source_capture.py
```

实际完整结果390 passed、8 skipped、1 xfailed；SDK专项19 passed。XML与原始stdout完整保存在`reports/m6g_execution_logs/full_tests.*`及`sdk_tests.*`，skip/xfail未删除或升级。原M6-D已知回退问题的xfail仍保留。

首次封存时执行的命令如下，使用排他创建；**已存在交付目录时不要原地重复执行**：

```bash
.venv-gpu/bin/python -m tools.m6g_release
.venv-gpu/bin/python -m tools.m6g_finalize
.venv-gpu/bin/python -m tools.m6g_release --seal
```

初次分析JSON在finalize前另存`analysis_v1_*.json`，保持数值和失败记录可审计。完整最终交付的文件SHA在`reports/m6g_release_manifest.json`，清单本身由最终Git提交保护，避免自引用SHA。不存在新的服务端build SHA，写null/NotRun。

## 构建条件审计与有限失败记录

官方CARLA0.9.16固定到`294096eb1c38eabf246e4f3a9cdab704e33a7f4c`；NVIDIA公开PhysX3.4参考固定到`5e42a5f112351a223c19c17bb331e6c55037b8eb`。所有已用源文件、URL、SHA/git blob在`m6g_sources`。保留上游许可证；不是从未授权镜像获取UE。

当前Git身份无法访问CarlaUnreal/UnrealEngine，官方API404；本地搜索未发现对应UE源码，UE4_ROOT未配置；cmake3.22.1、gcc11.4存在，匹配UE编译链缺失。空闲194,867,585,024bytes；容器实际内存上限50GiB、CPU quota14，宿主503GiB/128CPU不当作容器可用资源。磁盘超过官方130GB参考仍不足以证明完整构建峰值。

保留HTTPS代理提示、直连SSH publickey失败、现有Git配置Repository not found三种不同失败，最终以官方API和当前Git身份的实际不可访问为准，不把某次自选SSH配置失败当成全部身份结论。

额外公开源码查询中，误用`VehicleAIController.cpp`得到404；修正为`WheeledVehicleAIController.cpp`后成功。一次递归Git tree下载因IncompleteRead中止，随后仅查询具体contents目录获得正确文件；没有循环重试或下载引擎。补充失败说明在`m6g_execution_logs/retrieval_notes.json`，原错误清单保留。

结果为`Instrumentation_NotRun_BuildPrerequisiteFailed`。插桩patch、独立新二进制、行为等价性、内部时间单位验证、额外受控物理事件与服务端新增观测开销全部NotRun。未运行任何安装、make或服务端启动命令。

## 数据冻结及退出

已只读核验13,422份基线跟踪文件、50,867份继承外部资产、252份M6-F交付资产及4份0916二进制身份文件，无变化。范围只覆盖清单内资产，不声称穷尽未冻结外部文件。

封存`DynamicsObservabilityBaseline`后结束本轮深挖。不批准新控制器或Physics研究；路线管理、Oracle规划和Offline Shadow感知可以在原合同下分别推进，不代替真实停车、安全或严格20Hz验收。
