# M8-I1a 实现与复现

基线 `916ba75072259f17da725ddbc9c76f2f5fbeea13`；独立分支 `feat/m8i1a-startup-identity-diagnostic`。本轮实际执行仅为文件审计及 Python Mock/进程测试；真实 CARLA 启动 NotRun。没有修改旧 M8-I1 文件或默认驾驶系统。

## 模块

| 文件 | 职责 |
|---|---|
| `configs/m8i1a/protocol.json` | 固定地图、XODR、端口、180s/64MiB、有界被动确认与授权边界 |
| `tools/m8i1a_common.py` | 独立路径、SHA、排他写入、文件及目录 fsync |
| `tools/m8i1a_identity.py` | 只读身份采样、写后校验、Runner 结果和退出状态对账 |
| `tools/m8i1a_run_session.py` | 真实只读客户端入口；finally 后 `SystemExit(code)` |
| `tools/m8i1a_launch.py` | 独立授权凭据检查、单次目录锁、自有进程、截断日志上限、最终清理探针 |
| `tests/m8i1a_fakes.py` | 明确标记的 Mock CARLA 子进程，不导入 CARLA |
| `tests/test_m8i1a_startup.py` | A–G 及额外失败路径测试，真实 OS 退出码但合成 RPC |
| `tools/m8i1a_audit.py` | 静态审计、留档测试和 NotRun 报告生成；排他输出 |
| `tools/m8i1a_seal.py`、`tools/m8i1a_verify.py` | 新资产与启动前输入冻结、只读校验 |

## 身份与失败合同

identity_NNN.json 记录期望地图与实际完整名、Episode、Snapshot frame/elapsed/delta/platform time、Client/Server 版本、原 settings/raw string、天气、可用地图、实际 XODR SHA/字节数、完整启动 context。未知字段为 null，并记录具体读取异常。先关闭并 fsync 文件与目录，随后校验；存储失败直接拒绝，不能跳过记录继续运行。

实际 XODR 与本地资源 SHA 是不同证据。只读协议不保存训练 GT，不移动 Actor，不读取传感器或场景对象。Context 保存请求 CWD、Supervisor 实际 CWD、安装目录、完整 argv、源码 SHA；Supervisor 另外保存 /proc 中自有进程的实际 argv/CWD，不能将“请求 CWD”冒充 Shipping binary 实际 CWD。

成功必须是 rc0 + 同 run_id 业务成功 + 非空身份证据 + Runner cleanup + 最终 server cleanup + 无 Supervisor 错误。地图拒绝 rc12、XODR rc13、Map 不可读 rc14、普通 Runner 异常 rc20、终止 rc21、清理失败 rc23。完整枚举见 [退出合同](../reports/m8i1a_exit_status_contract.json)。SIGKILL 可能没有 Session Result，监督器按非零／缺失结果拒绝，保留此前身份，不制造 finally 日志。

Supervisor 不因 returncode0 或 failure=null 单独放行；业务、退出码或 cleanup 矛盾即 `StatusPropagationNoGo`。日志通过有界管道读取，不任由文件无限增长；工作145s、清理25s，为180s上限预留时间。真正的 CARLA 进程编排、最大日志及超时行为未实测，不能把 Mock 对账测试描述为全 Supervisor 实时验证。

## 已执行命令与安全离线复核

```bash
.venv/bin/python -m tools.m8i1a_audit
```

此命令已执行一次；使用排他目录，不应原地重跑。10 个子进程案例、25 项测试和 SHA 索引位于 `/root/gpufree-data/hs-drive-experiments/m8i1a-offline-evidence/`。其中全部 World 身份均为 SyntheticMockCARLA，不能用于实际地图结论。

以下复核不启动 CARLA，也不覆盖封存报告：

```bash
.venv/bin/python -m pytest tests/test_m8i1a_startup.py -q
python3 -m tools.m8i1a_verify
python3 -m tools.m8i1_verify
git diff 916ba75072259f17da725ddbc9c76f2f5fbeea13 --name-status --diff-filter=MDTR
```

最后一条应为空。测试默认使用新临时目录；不要设置已封存的 `M8I1A_TEST_OUTPUT` 重跑。测试产生的 Python 进程可被终止，未运行 UE4。

## 待单独批准的运行入口

当前不存在授权凭据，真实外部目录 `/root/gpufree-data/hs-drive-experiments/m8i1a-startup-identity` 也未创建。任务书第六节明确“以下步骤必须获得新的独立运行授权”；完成离线检查不自动提供该授权。

获用户明确批准后才能记录一份真实批准凭据：scope=`M8-I1a_OneReadOnlyStartup`、approved=true、max_launches=1、用户批准消息引用、冻结 protocol SHA；不能伪造批准引用。启动器同时校验预冻结源码、报告和地图资源。凭据是工作流记录，不是抵抗恶意代码的安全沙箱。

```bash
# 仅在上述独立授权后；本轮未执行：
.venv/bin/python -m tools.m8i1a_launch --authorization /path/to/actual_approval_receipt.json
```

固定原 server argv 与当前仓库 CWD。匹配须两次同 Episode/Map/XODR/settings 且不倒退的被动 Snapshot；第一次地图不符立即留证拒绝。版本连接最长90s、身份窗口最长30s，不换图、不改World、不创建TM、不Tick。异步引擎可能自行推进；Owner Tick0不代表引擎Frame不变。

启动器创建一次性目录作为额度消耗记录，任何失败都不能删除后重试。即使通过也立即关闭服务器，不进入交通采集。真实结果应追加新报告或另立已授权执行归档，不能覆盖本轮 NotRun 交付或旧 I1 失败。
