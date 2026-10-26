# M8-J1 维护者询问记录

审核日期：2026-10-26。状态 **FinalDraftReady / NotSent / PermissionClarificationUnresolved**。

本轮用户明确规定对外发送需独立授权；目前没有该授权。已完成文本精简，没有创建GitHub Issue、邮件或HF公开讨论；没有待回复的已发请求。`PendingSendAuthorization`只表示发送步骤尚未获准，不表示维护者正在处理。

| 字段 | 记录 |
|---|---|
| 拟议渠道 | 官方仓库GitHub Issue，由用户后续决定是否授权 |
| 官方对象 | [Thinklab-SJTU/Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) 仓库维护者；数据仓库 [rethinklab/Bench2Drive](https://huggingface.co/datasets/rethinklab/Bench2Drive) |
| 实际发送渠道 | NotSent |
| 发送时间 | NotAvailable — 未发送 |
| 公开问题链接/发送凭证 | NotAvailable — 未创建 |
| 实际发送内容 | NotAvailable — 下方仅为最终草稿 |
| 对方答复/答复人角色/适用资产 | NotAvailable — 无本轮联系产生的答复 |
| 自动回复或第三方解释作为授权 | 否 |
| 草稿来源 | [M8-J0许可矩阵](m8j0_license_scope_matrix.md)；原文件与SHA保持不变 |

拟议标题：**License clarification for one legacy Bench2Drive Mini archive**

最终英文询问正文（未发送）：

> We are reviewing one legacy Mini archive in `rethinklab/Bench2Drive`: `DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz` (280176944 bytes; SHA256 `b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2`), listed at GitHub commit `7ec25d1c9f7522d923ce5f3420986cef1cb2d956`.
>
> The GitHub LICENSE/README state CC BY-NC-ND 4.0, while the HF data card states Apache-2.0. Which license, exact HF revision, and any third-party restrictions apply to this archive?
>
> Could you clarify the conditions for: (1) noncommercial local download and private one-frame review; (2) noncommercial research-model training; (3) publishing aggregate metrics and a few annotated images; (4) releasing model weights or derived annotations; and (5) redistributing original or processed data?
>
> Our immediate proposal is only (1), with no training or redistribution. Please also identify any official alternative access route for this same archive, if available. A versioned asset-specific statement would help us preserve the applicable conditions and attribution.

今后若独立授权发送，应另行追加发送渠道、UTC时间、完整原文、公开链接/凭证和维护者答复，不回写本轮NotSent事实。仅有沉默、自动回复、非维护者意见或泛指其他发布物的答复，不能关闭目标片段的许可门禁。
