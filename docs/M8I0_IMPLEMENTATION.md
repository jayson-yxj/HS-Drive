# M8-I0 证据审查与复现

本轮只增加m8i0报告、文本证据和只读审计工具，无主系统实现改动。基线 `69ce3158ab692387e615a781deb41aae0fb61499`。无额外Python安装；所有脚本用标准库，可在仓库根目录执行。

## 已执行

1. 创建 `feat/m8i0-existing-scenario-data-review`，读取指定M8-F/G/H0/H1架构、H1原始报告、M8-B质量结果。
2. `python3 -m tools.m8i0_fetch_sources configs/m8i0/sources.json`，再按依赖顺序审查source_details/source_schema/source_final/source_runtime五份清单。HTTPS证书验证开启；每文件最多2MiB、每批声明上限25MiB，只允许固定官方主机文本。保留失败，无后台下载/无限重试。GitHub Contents响应解码后存原始源码文本和Git blob/SHA256；未执行下载的代码。
3. `python3 -m tools.m8i0_audit` 验证260条G/H0/H1历史记录、官方源码内容与本机示例一致性；从固定XML生成三类场景目录，从AST提取六摄定义，未import外部框架。
4. 逐项填写GT能力矩阵和未取得字段，完成唯一优先/备用及有限Pilot提案。实际CARLA/PETR/GPU/数据样本运行均NotRun。

## 只读复核

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
python3 -m tools.m8i0_verify
git diff --name-status 69ce3158ab692387e615a781deb41aae0fb61499 --diff-filter=MDTR
```

第二条应为空：基线已有文件未改/删。验证器只读检查本轮release manifest、历史SHA、数据资格与无运行约束，不连接CARLA或网络。结果验证不是仿真Smoke/模型质量测试。

若需重新生成派生JSON，在单独临时worktree执行以下命令；输出含新的审计时间，不应覆盖已封存发布报告：

```bash
python3 -m tools.m8i0_audit
```

下载工具默认校验已有证据，不会覆盖或自动重试失败。未来复查移动中的HF卡或新提交应使用新阶段/新证据路径。本轮API失败与网页可读并存，不能只用网页检索结果表示本机文件可下载。

`m8i0_release_manifest.json`记录本轮源码/配置/报告/公共证据SHA；自身与最后验证日志不自哈希，Git提交固定其版本。原M8-H1外部资产仍由历史manifest验证，不复制、不改名。完整可重现运行环境需历史外部资产路径可用，迁移机器缺失应报错而非忽略。

下一Pilot没有可执行场景命令或采集代码。用户单独批准后才能实现提案内的Session owner及Evaluator；当前复现命令仅审查。
