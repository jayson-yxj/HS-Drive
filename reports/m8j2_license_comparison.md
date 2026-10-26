# M8-J2 官方许可对照

实际检查：2026-10-26。**LicenseConflictUnresolved / LicenseScopePartial**。目标仍为旧Mini片段 `DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz`。本轮访问恢复没有转化为训练使用授权。

| 层级 | 证据 | 结论 |
|---|---|---|
| GitHub代码与仓库资产声明 | 固定 `7ec25d1c9f7522d923ce5f3420986cef1cb2d956` 的LICENSE为CC BY-NC-ND4.0，README声明覆盖资产与代码、另有说明除外。M8-J0/J1取得的原字节本轮重新核验SHA。 | 不是单纯的“只管代码”声明，不能直接排除数据资产。 |
| HF数据集仓库声明 | 本服务器通过代理取得固定 revision `ce56cfbc26edf7796586e26fd44644f0ac2f6bcc` 的README，HTTP200，2096bytes；YAML和正文均声明Apache-2.0。 | 比此前缓存网页更强，且已绑定与目标文件相同的仓库快照；但未解释与GitHub声明的关系。 |
| 指定压缩包与第三方条件 | API确认文件属于该快照，名称/大小/LFS SHA均匹配。返回的仓库文件列表没有LICENSE/NOTICE命名条目，未查看包内任何文件。 | 没有完整的逐资产条款、第三方NOTICE或例外证据；不能从没有独立LICENSE文件推断没有附加限制。 |

原始来源：[固定GitHub LICENSE](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/LICENSE)、[固定GitHub README](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/README.md)、[固定HF数据卡](https://huggingface.co/datasets/rethinklab/Bench2Drive/blob/ce56cfbc26edf7796586e26fd44644f0ac2f6bcc/README.md)。SHA与服务器证据见[许可JSON](m8j2_license_evidence.json)。

本轮GitHub新请求经代理返回403，JSON明确为匿名API rate limit；这不是TLS失败、数据访问禁令或许可拒绝。没有切换账号、出口或重试绕过限流；固定提交的历史官方证据仍可复核，不谎称本轮重新HTTP200取回GitHub声明。外部浏览服务读取仅作补充，正式HF结论依据服务器原始响应。

M8-J0/J1的使用条件分层不变：非商业内部复制与单帧审核有条件性支持；不能将CC ND解释为禁止所有内部研究/训练。公开权重、衍生标签、含源图像的标注图及原始/处理数据再分发需要逐项澄清。聚合统计指标与图像分享不等同。未知第三方限制不能自动忽略，也不直接等同全面禁止。

**下一步仅保留维护者澄清材料，不发送。** 复用[已定稿询问](m8j1_maintainer_contact_record.md)，可附加以下实际事实：

> We can now access the official repository anonymously at revision `ce56cfbc26edf7796586e26fd44644f0ac2f6bcc`. The archive metadata matches the published 280176944-byte size and SHA256. The README at this same revision states Apache-2.0, while the fixed GitHub LICENSE/README state CC BY-NC-ND 4.0. We have not downloaded the archive. Please clarify which asset-specific terms and third-party restrictions apply to the previously listed uses.

没有新Issue、邮件、公开讨论、自动回复或维护者答复。状态仍为**NotSent / Unresolved**。不因网络成功批准下载、训练、模型适配或视觉控制。
