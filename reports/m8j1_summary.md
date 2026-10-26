# M8-J1 — Bench2Drive许可澄清与官方访问路线闭合

实际检查日期：2026-10-26。基线 `307f4fb34b501c6400f20e4e437963429527a333`；独立分支 `feat/m8j1-bench2drive-permission-access-closure`。

**最终选择B：`LicenseOrAccessStillUnresolved`。** 本轮限定核实已完成，但许可及访问缺口仍在；停止继续建设下载工具，不自动申请文件下载。不将未解决问题包装成“条件已闭合”，也没有证据选择资产永久NoGo的C。

| 条件 | 本轮实际结果 |
|---|---|
| 历史冻结 | M8-J0 44个交付文件SHA与原状态保留；经原验证器再次核验306项历史资产。旧文件修改0。 |
| 许可 | 固定GitHub LICENSE/README通过服务器API200重新读取，原文字节/SHA完全吻合；HF浏览卡仍声明Apache。未取得此片段的正式资产例外或协调说明。 |
| 维护者 | 最终英文询问已精简保存。`NotSent`，无发送时间、公开链接或答复，不声称已进行许可沟通。 |
| GPU服务器DNS/TCP | HF域名返回IPv4/IPv6地址；本批次仅测试一个IPv4地址，5秒TCP超时，未进入TLS。地址正确性和失败根因未独立证明，不认定DNS污染或永久封禁。 |
| GPU服务器HTTPS | HF仓库API与唯一目标HEAD均curl28连接超时，无HTTP状态；GitHub TLSv1.3验证及3次API GET正常。 |
| 其他环境 | 浏览研究服务可读缓存HF卡和讨论页，目标blob元数据页报工具错误；不能转写为GPU服务器访问成功或文件404。 |
| 对象身份 | 官方清单的文件名、280176944bytes及SHA保持确认；当前托管修订、LFS/Xet、Content-Length和ETag仍未知。 |
| 其他官方路线 | 固定README列Baidu Base分享，下载脚本列HF CLI；只是来源说明，未验证该片段在其他入口可取得，不使用第三方镜像。 |

唯一目标继续为 `DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz`，预期SHA256 `b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2`，官方清单提交 `7ec25d1c9f7522d923ce5f3420986cef1cb2d956`。未改选其他片段。

本次HF失败是**连接超时**，M8-J0是连接拒绝，两次事实独立保留。curl的`ssl_verify_result=0`在从未建立TLS时不代表证书验证成功；也没有401/403、404或服务端配额证据。批次为06:51:55–06:52:18 UTC：2个主机的有界DNS/单地址TCP/TLS层诊断，5次HTTP请求（3成功、2失败），保存74,176bytes官方元数据正文，0重试、0Range、0载荷GET。条件性后续HF请求因入口失败而跳过。准确逐请求时间和指标见[访问JSON](m8j1_official_access_evidence.json)。

许可核实区分三层：[固定GitHub许可](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/LICENSE)、[HF卡片](https://huggingface.co/datasets/rethinklab/Bench2Drive/blob/main/README.md)、未检查的压缩包/第三方条款。未选较宽松声明；也未将CC ND解释为禁止所有非商业内部研究。各用途的未决条件及未发送询问见[许可澄清](m8j1_license_clarification.md)与[联系记录](m8j1_maintainer_contact_record.md)。

没有实质许可/可执行访问进展，因此未新增一帧审核申请或数据合同实现。M8-J0原相机K/T、缺失时间、静态身份/实例关联和Unknown限制保持；单帧Schema并非训练或驾驶认证。没有CARLA、解压、GPU、推理、训练、默认模型/控制修改或main合并。

5项合成请求边界测试及离线证据验证通过，只验证审核工具和归档一致性。`LicenseConflictUnresolved / OfficialMetadataUnavailable / PayloadNotVerified / TrainingReadinessNoGo`仍保留。[验收](m8j1_acceptance.json)、[明确止损与下一决策](m8j1_source_decision.md)、[架构决策和复核命令](../docs/M8J1_ARCHITECTURE_DECISION.md)。
