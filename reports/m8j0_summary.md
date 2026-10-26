# M8-J0 — Bench2Drive单片段许可与访问条件评审

2026-10-26，基于 `fe8db837d18f033e446475d0bf48534eb54bc0f3`，独立分支 `feat/m8j0-bench2drive-license-access-review`。

**评审完成；当前决策 `AccessOrLicenseNoGo`。暂不具备单片段实际下载申请资格。** 文件身份的官方清单部分明确，许可适用差异和服务器访问条件尚未闭合。不是文件不存在或所有内部研究均被禁止的结论。

实际服务器共7次有界HTTP尝试，5个GitHub元数据GET成功、2个HF请求在连接阶段失败；保存39,060bytes公开响应正文。HF失败分别是仓库元数据API和唯一目标HEAD，均curl7 Connection refused，无HTTP状态，故不是401/403、404、TLS证书失败或文件配额证据。没有连续重试、Range、目标GET、载荷下载或解压。另有浏览研究服务读取公开卡片、许可与仓库页面；这些可缓存的网页证据不能证明服务器网络畅通。[请求原记录及时间](m8j0_official_access_probe.json)。

固定官方源码 `7ec25d1c9f7522d923ce5f3420986cef1cb2d956` 的 [Mini清单](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/docs/bench2drive_mini_10.json)重新取得并与历史冻结字节一致，确认唯一目标：

- `DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz`
- 280,176,944 bytes。
- SHA256 `b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2`。
- 官方下载脚本指向 `rethinklab/Bench2Drive` 根目录。它是旧Mini/Base资产，不是0.0.4新增数据、独立Depth包或Dev10闭环路线。

上述是**预期**文件身份，不是取得文件后的校验结果。浏览页给出仓库/README修订候选 `ce56cfbc26edf7796586e26fd44644f0ac2f6bcc`，尚未将目标对象与此修订、LFS对象哈希绑定。[详细身份](m8j0_official_asset_identity.json)。

固定GitHub许可为CC BY-NC-ND4.0，HF YAML与正文为Apache-2.0。HF声明日期早于GitHub许可变更，但这不能确定资产例外或消除冲突。CC并非全面禁止非商业内部复制/训练；对外图像、衍生标签、权重和数据分享仍须按用途核查。已提供逐用途矩阵和维护者询问草稿，未发送。[许可矩阵](m8j0_license_scope_matrix.md)、[原始来源与检索边界](m8j0_license_conflict_evidence.json)。

源码可支持未来单帧研究：六摄1600×900、前/侧70°与后110°FOV，K、cam2ego、world2cam及Actor Box字段有出处。与HS安装参数不同，旧包PNG Depth不能混用新NPZ合同。JPEG20可能削弱远场目标；静态车Box与身份关联仍有歧义。原始逐传感器Frame/Timestamp未在所查保存字段中保留，同名文件不能证明原始时间一致。所有字段独立分级，全部载荷状态为 `PayloadNotVerified`。[Schema](m8j0_source_schema_contract.json)。

M8-I1b历史306项资产再次核验通过，结论 `StartupRootCauseUnresolved / NoRunMatchedUELog / TrainingReadinessNoGo`保留。M8-I1的版本RPC成功后地图拒绝，与M8-I1a的31次版本RPC超时仍独立记录；单次启动授权已消耗，三层来源勘误原文未改。新增冻结索引369条路径记录对应357个真实文件（含路径别名），没有冒充369份独立历史样本。补充日志勘误清单也已核验。[冻结](m8j0_historical_freeze.json)、[离线验证](m8j0_verification.json)。

8项离线请求边界测试通过，包括禁止目标GET/其他包/未核实域名、HTTPS及URL脱敏。测试仅验证审查工具边界；不认证数据可用、网络稳定、训练或驾驶。未启动CARLA、GPU、PETR或训练，未改默认模型、M8-B Adapter、控制及历史资产。

下一步停止自动推进。先解决官方资产许可说明、本机HF元数据可达性及目标文件修订；条件闭合后才能单独申请一个文件和一帧Schema审查。未来读取上限与失败即停策略已写入[资源计划](m8j0_single_clip_resource_plan.md)，当前不执行。[A–G完整决策](m8j0_next_action_decision.md)、[实施与复核命令](../docs/M8J0_IMPLEMENTATION.md)。
