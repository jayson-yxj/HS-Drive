# M8-J1 架构与数据路线决策

日期：2026-10-26。基线 `307f4fb34b501c6400f20e4e437963429527a333`。独立分支 `feat/m8j1-bench2drive-permission-access-closure`。

选择 **B — LicenseOrAccessStillUnresolved**。完成本轮一次有界核实后停止Bench2Drive自动推进，不继续下载器工程。已有证据不足以发出单片段审核申请，也不足以证明目标资产永远不能满足项目条件。

决定依据：固定GitHub声明与HF资产卡未协调；没有维护者授权回复，问题也尚未发送。GPU服务器HF只完成DNS返回、未建立所选TCP连接，更未到TLS/HTTP；官方对象身份无法定位。另一环境缓存卡片可读不能填补这些缺口。官方Base百度入口是记录中的可能访问方式，没有片段级可执行证据，不据此绕过本轮访问门禁。

维护者询问已完成，可审查最终原文，但本轮不要求用户批准一个未经准备的动作，也不将审核草稿视作已发送。对外创建Issue/邮件仍需用户独立明确授权；任何答复必须绑定此文件、版本和用途。当前状态为NotSent而非AwaitingMaintainerReply。

不更新单帧Schema申请，不移植Adapter，不改变K/T/时间/身份合同。`PayloadNotVerified / TrainingReadinessNoGo`保持；M8-I1b根因未明、M8-J0许可及访问NoGo、已消耗的CARLA启动授权全部保留。

下一决策可另行限定为“有明确获取条件的小型公开数据源评审”，优先考虑nuScenes Mini等与PETR原始训练分布相近的候选；本轮没有开始其许可/网络/支持量评估，未批准下载或训练。不能通过换片段规避未决条件，也不恢复CARLA启动修补循环。

实现范围是一次性诊断记录器，不是数据下载工具：`tools/m8j1_metadata_batch.py`仅使用GET公开元数据与唯一文件HEAD；单批次独占开始标记，失败后也不可自动再运行；没有代理/DNS修改、自动重试、重定向跟随或载荷读取。已执行的批次关闭，未用完的上限不构成后续额度。具体上限见`configs/m8j1_review_protocol.json`；实际5次HTTP、2个主机网络分层测试、1次浏览调用含3个官方元数据页面。

离线复核命令（不触发网络）：

```bash
python3 -m unittest tests.test_m8j1_metadata_boundaries -v
python3 -m tools.m8j0_verify
python3 -m tools.m8j1_finalize verify
git diff 307f4fb34b501c6400f20e4e437963429527a333 --name-status
```

`tools/m8j1_finalize build`只从已有证据产生报告，输出采用独占写入，不覆盖历史；已存在报告时不重复运行。实际联网命令是`python3 -m tools.m8j1_metadata_batch`，仅作为已执行记录，**本阶段不得再次执行**。每条URL、方法、时间、退出码、TLS配置及source SHA都在`reports/m8j1_sources`。`reports/m8j1_release_manifest.json`绑定报告、源码、配置及日志；提交SHA由Git包含关系提供。

证据分级：服务器实测、独立浏览缓存、离线源码/清单核验、合成边界单测分别标记；载荷、单帧Schema、模型及闭环均NotRun。原样保留官方证据，不将网络问题写成许可禁止，也不将无答复写成许可授予。

关联：[总结](../reports/m8j1_summary.md)、[验收](../reports/m8j1_acceptance.json)、[最终源决策](../reports/m8j1_source_decision.md)。
