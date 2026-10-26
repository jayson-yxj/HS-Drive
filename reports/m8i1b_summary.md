# M8-I1b：CARLA启动与RPC离线根因审计

**决策 B：`StartupRootCauseUnresolved`。停止无明确证据的CARLA启动修补，建议下一项转为Bench2Drive单片段许可与数据访问条件评审。** 本轮没有启动服务器、连接CARLA、采集或训练，也没有执行下一项评审。

基线 `5c14c347de2b183d59d7bd3f604ec45d0b602da6`；分支 `feat/m8i1b-offline-startup-forensics`。M8-H1、I1、I1a五份相关manifest共306个去重资产逐项SHA复核；原始运行、执行代码、授权记录和失败输出保持不变。

## 日志取证结果

| 来源 | 本轮结果 | 能说明什么 |
|---|---|---|
| 安装树／执行用户的UE日志位置 | `NoRunMatchedUELog` | 未找到匹配目标运行的独立UE加载/事件日志，不可补造 |
| I1／I1a自有stdout | 已关联、SHA及mtime留存 | 有引擎头和退出文字，没有加载地图、逐行时间戳 |
| 现存CrashContext | 10月19日、PID799952、port2000、Low | 明确属于其他运行，不用于本轮根因 |
| CrashReportClient配置 | 启动窗口附近mtime相关，但无PID/错误/地图 | 配置写入不是崩溃证明 |
| Host内核ring／journal | 目标I1/I1a窗口无相关错误；journal无存储 | 不支持明确环境错误，也不能证明环境绝无错误 |

每个候选文件的路径、SHA、mtime/ctime、UID及归属可信度见[日志库存](m8i1b_runtime_log_inventory.json)。只读命令、UTC窗口过滤结果和原始dmesg快照保存在独立外部审计目录；不引用其他轮次崩溃记录作为本次事实。

## 启动链与对照

| 阶段 | 可支持的最高状态 | 不能延伸的结论 |
|---|---|---|
| M8-H1 | 最终WorldMapIdentityAvailable、MapXODRVerified | preflight允许load_world，不能证明初始就为Town02 |
| M8-I1 | CARLAVersionRPCReady；读取地图后拒绝但名称未保存 | 不能推测实际地图，XODR验证未到达 |
| M8-I1a | ServerProcessCreated、ShippingProcessAlive、TCPListening | 版本RPC31次超时，无World/Map/XODR证据 |

完整argv、CWD、执行用户、环境、地图加载方式、就绪条件、Client版本、端口、单次超时、等待预算、资源和关闭方式已逐项对比。每项标明实际记录、已执行源码、文档命令、推断或缺失。H1的60s RPC、I1新建Client重试与I1a复用Client等差异确实存在，但没有证据证明它们导致I1a失败。I1监督器同样用2s RPC却成功，不能仅凭I1a的31次2s超时就认定应加大timeout。

本轮不把两次失败统一解释为地图加载异常，不把TCP连接当RPCReady，不把实际GPU利用率或退出EOF当错误来源。详见[路径对比](m8i1b_success_failure_launch_diff.json)、[六态链](m8i1b_rpc_readiness_evidence.json)与[假设审查](m8i1b_root_cause_hypotheses.md)。

## 授权来源勘误

用户补充了外部对话转录：**外部user申请 → 外部assistant批准 → user转述给Codex执行**。已追加[勘误](m8i1b_authorization_record_erratum.md)，原授权文件SHA仍为 `6abee8f1a7ab7fbec4f9134eba616aea133d63b2556d1eda735642bc1bb39023`。

外部作者与本地转述角色分开记录。外部转录为用户提供、未有平台导出独立认证；消息ID、精确平台时间和分享链接均未补造。旧凭据中的“批准了”作为本地实际接收文字保留，不另造一条外部user批准。此次勘误不增加启动额度。

## 复核与下一步

本轮是证据审计，没有新的仿真测试。验证包括历史SHA、JSON/源码语法、文档链接、三条状态链与最终B决策一致性；可用以下命令离线复核：

```bash
python3 -m tools.m8i1b_verify
python3 -m tools.m8i1a_runtime_verify
git diff 5c14c347de2b183d59d7bd3f604ec45d0b602da6 --name-status --diff-filter=MDTR
```

最后一条应无输出。原始新取证目录：`/root/gpufree-data/hs-drive-experiments/m8i1b-offline-forensics`。采集脚本采用排他输出，不应覆盖该目录重跑；使用verify做无副作用复核。

不建议新的CARLA启动申请。下一阶段仅建议审查Bench2Drive一个官方片段的许可适用范围与数据访问条件；需要独立任务范围，不自动下载或运行。原I1、I1a的NoGo、模型/Occupancy/视觉安全门禁均不升级。详见[验收](m8i1b_acceptance.json)及[架构决策](../docs/M8I1B_ARCHITECTURE_DECISION.md)。
