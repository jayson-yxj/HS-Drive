# M8-I1 实现、执行版本与复现

基线：`502f0701025f2de5a8f2cc1ecbdcd4b3cacd2b6a`。分支：`feat/m8i1-traffic-manager-development-pilot`。唯一实际服务器运行提交：`2a5edce0a78fd0e8d08fd96dd6c27015a8dd5490`。状态：PilotNoGo，地图不匹配后结束，尚无交通、同步采集或可评价 GT。

## 文件职责

| 文件 | 职责与验证边界 |
|---|---|
| `configs/m8i1/protocol.json` | 运行前冻结的原 Rig、种子、预算与拒绝规则 |
| `tools/m8i1_launch.py` | 一次性目录锁、自有服务器、时间／磁盘监督、进程组清理；实际运行过失败路径 |
| `simulator/m8i1_session.py` | 唯一 Tick Owner；地图检查实际失败；其余创建／采集路径未实测 |
| `simulator/m8i1_records.py` | Source OBB、原 Physics、Transform 的序列化；不改 Physics |
| `datasets/m8i1_contract.py` | 原始帧相位调度与严格 RGB 加载器；合成单测通过 |
| `tools/m8i1_isolation_probe.py` | 合成文件读取追踪、路径攻击拒绝及 OS 权限探针；已实际执行 |
| `tools/m8i1_closeout.py` | 从保留的失败证据生成 NotRun 报告；追加式，不允许覆盖已生成文件 |
| `tools/m8i1_seal.py` | 新交付及外部日志 SHA 清单；追加式 |
| `tools/m8i1_verify.py` | 离线核验交付、执行源码、旧资产、失败状态；不连接 CARLA |

创建路径使用一次原生 Spawn 尝试、首个有效车辆作为 Ego、自然落稳再启用 TM。Walker 由独立 AI Controller 控制，不计为 TM 车辆。没有调用旧定制放置器，没有主动忽略交通规则或改车辆物理。24 路 RGB/Depth/Instance/Semantic 按实际 frame 和 timestamp 同步；0.1s 相机相位从回调识别，不假定偶数帧，不最近邻补齐。

以上为实现意图和已编写路径，不是实际采集成绩。尚需实际验证 Actor 生命周期、动态相机闭合、原始标签编码、被动渲染身份关联和完整清理。`protocol.identity` 仅为预声明规则，尚无已验证的渲染身份评价器。没有把缺测背景输出成 Free。

## 输入与 GT 合同

`RGBInferencePacket.m8i1.v1` 仅含 source、calibration SHA、camera order、image size、frame、timestamp、Identity 增强以及六摄 RGB 相对路径／SHA／K／T。`RGBInferenceLoader` 精确检查顶层和相机键、路径、图像大小、坐标矩阵和时间，并返回六摄 RGB 数组。它不加载 Depth、Instance、Semantic、Actor Transform 或 Box。

`EvaluatorSidecar.m8i1.v1` 独立保存特权 Snapshot、Source OBB、原始非 RGB 载荷、Ego/相机世界变换、地图角点查询与静态库存。Actor 身份必须与 world episode 组合；Source Box 是 Actor Transform 与本地 Box 的合成，不是精确 Mesh。CARLA world 为 x 前/y 右/z 上；HS Ego 为 x 前/y 左/z 上；光学为 x 右/y 下/z 前。

当前只在合成文件上验证白名单和独立 Unix 权限，不声称真实运行目录已建立用户级隔离。正式 RGB 推理不允许使用 GT 可见性／匹配 Mask；后续评价才可使用。

## 实际执行与失败留存

实际执行命令为下列一次性启动；**此处仅记录历史命令，不授权重跑，也不能删除目录锁重跑**：

```bash
.venv/bin/python -u -m tools.m8i1_launch
```

完整 server/runner argv 保存在外部 `execution_commands.json`。服务器以 ubuntu 用户执行官方 `CarlaUE4.sh`，传入 Town02、Epic、RenderOffScreen、RPC2100；原安装和主 GPU 环境未修改。期望版本检查通过，地图名称检查失败，未进入地图 XODR、World 设置、TM、Spawn 或 Tick 阶段。原始地图文件 SHA 已在运行前核验，但不能代替实际加载地图身份。

外部目录：`/root/gpufree-data/hs-drive-experiments/m8i1-traffic-development/`。
保留 `launch_once.json`、命令、server/session stdout、事件、session_result、cleanup、资源采样、supervisor_result、事后 OS 探针及静态启动脚本检查。没有图像；不提供虚构场景可视化。`session_result.failure` 是业务失败事实，退出码 0 不是通过；原执行代码冻结，未事后改写它的错误处理。

运行前清单 `reports/m8i1_preflight_freeze.json` 中的 `execution_commit` 指冻结时 HEAD（基线），同时包含当时未提交的新文件 SHA；它们随后提交为上述执行版本。事后报告中的同名字段亦是生成时 HEAD，可能含未提交的离线工具；对应 `source_sha256` 和最终 release manifest 才完整限定各版本。实际 Session 的源码哈希均能从执行提交取回。最终补充的第 9 条地图拒绝单测与预运行 8 条版本不同，不能拿最终文件哈希冒充预运行测试。

## 可安全复核的离线命令

以下命令不启动服务器、不运行模型，也不会覆盖原报告：

```bash
.venv/bin/python -m unittest tests.test_m8i1_contract tests.test_m8h1_evidence -v
.venv/bin/python -m pytest tests/test_frame_sync.py tests/test_reference_geometry.py -q
python3 -m tools.m8i0_verify
python3 -m tools.m8i1_verify
git diff 502f0701025f2de5a8f2cc1ecbdcd4b3cacd2b6a --name-status --diff-filter=MDTR
```

结果：21 + 17 = 38 项不同测试通过；历史足迹 5 条另行通过；旧 127 项 I0 清单及 260 项封存记录复核。最后一条应无输出，表示基线已有文件未修改。完整日志位于 `reports/m8i1_logs/`。

隔离探针和报告生成器已各执行一次，使用排他创建保护原结果。若仅需重新验证，使用上面的 verifier；不要为重跑生成器删除记录。需要重新生成离线研究产物时应在独立副本输出到新路径，而不是修改本次封存。

下一项启动排查需单独授权：先保存实际地图、Episode、frame、版本和启动完成证据，再考虑参数修正；修正 Session→Supervisor 失败状态传播。任何下一次交通采集仍需新预算、源码冻结及明确批准，不继承本次已使用的一次启动额度。
