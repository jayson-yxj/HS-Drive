# M8-I1a 已授权执行记录

本文件追加于原离线实现和架构决策之后，不替换其历史快照。真实执行源码为 `2cc22439587976d87550bf411e22f625d30508d0`，运行期间源码和冻结参数未修改。真实运行结果是 `ClientConnectionFailed`，不是地图匹配通过或地图不符。

## 授权和实际命令

授权来自当前对话中用户回复“批准了”及五项执行要求。授权 JSON 保存原文、相对上下文说明、记录创建时间、scope、执行提交和 protocol SHA；没有可见消息ID，字段为null。批准仅覆盖一个只读服务器进程，不覆盖第二次启动或TM。

以下为**已经执行过**的命令，不授权再次执行：

```bash
.venv/bin/python -m tools.m8i1a_launch --authorization /root/gpufree-data/hs-drive-experiments/m8i1a-approved-execution-evidence/authorization.json
```

实际 Supervisor 退出码1，Runner退出码10。单次目录锁已创建，不可删除它绕过额度。完整server/runner argv、请求CWD、实际进程 `/proc` 信息、31次连接异常、Session、Runner cleanup、Supervisor与最终清理过程均保留。

## 产物路径

- `/root/gpufree-data/hs-drive-experiments/m8i1a-startup-identity/`：唯一启动原始输出。
- `/root/gpufree-data/hs-drive-experiments/m8i1a-approved-execution-evidence/`：真实授权、执行前SHA/端口检查、运行中OS探针、独立事后OS探针及命令退出码观察。
- `reports/m8i1a_execution_*.json`：追加实测结论；`m8i1a_execution_summary.md` 是当前状态入口。
- `reports/m8i1a_execution_manifest.json`：新增代码、报告与上述原始输出 SHA。

没有 `identity_000.json` 是真实结果：版本RPC未成功，代码没有调用get_world/get_map；不得补造这一文件或用本地Town02包SHA替代实际世界SHA。执行版本的持久化校验逻辑与25项测试仍然有效，但实际身份记录路径未验证。

## 验收与资源

Runner业务失败、进程非零、Supervisor非零一致，实际路径 `ClientConnectionFailed → 10 → 1 → StartupDiagnosticNoGo`。`StatusPropagationGo` 仅表示此失败路径传播正确，不表示启动成功。清理21次探针保留瞬时未退出状态；最终OS探针及独立复核确认进程组73956、73957消失，2100–2102/8100关闭。

没有创建本任务Actor或Sensor，不存在待销毁的本任务Actor ID。由于RPC失败，没有完成World级对象库存或设置读回；不将OS清理证明夸大为逐Actor审计。未调用World/TM设置或车辆控制；引擎自己的异步运行不等于Owner Tick。

## 只读复核命令

```bash
python3 -m tools.m8i1a_runtime_verify
git diff 2cc22439587976d87550bf411e22f625d30508d0 --name-status --diff-filter=MDTR
```

第一条核对实测产物、执行提交源码SHA、旧I1/I1a封存SHA、批准原文记录、退出码和清理证据，不连接CARLA。第二条应为空，全部新增产物追加保存。

原 `tools.m8i1a_verify` 刻意断言真实启动目录不存在，用于批准前离线验收。现在已合法执行，它的“NotRun”前提不再成立；不修改旧校验器伪造未运行，请使用新增runtime verifier。历史文件SHA仍按原manifest核验。

本次诊断不是Traffic Pilot、训练或驾驶验收。下一次真实服务器操作必须有新的独立范围与授权；不根据本次TCP可达、默认配置或高GPU利用率擅自选择新地图、端口、种子或参数。
