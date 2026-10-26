# M8-I1a 授权来源追加勘误

**勘误：原凭据的 `user_message_text` 记录了 Codex 收到的用户转述内容，不能据此将其中批准文字的外部原作者归类为 user。** 根据本轮用户补充的外部对话转录，原始授权申请由 user 提出，原始批准由 assistant（ChatGPT）作出，再由用户交给 Codex 执行。

| 层次 | 角色与内容 | 证据及可信度边界 |
|---|---|---|
| 外部 ChatGPT 原始申请 | user；提交2cc2243、25项测试后请求一次只读启动授权 | 用户本轮提供的转录，非独立平台导出 |
| 外部 ChatGPT 原始批准 | assistant；“批准。M8-I1a 可以执行一次受限的 CARLA 只读启动诊断。”及限制、五项要求 | 同上；不得将该助手自述“我已核对提交”当成独立核验记录 |
| Codex接收与执行指令 | 本地角色user；“批准了”及五项执行要求 | 本会话可见消息及已封存凭据；这是转述提交者，不是外部原作者 |

原始授权凭据路径：`/root/gpufree-data/hs-drive-experiments/m8i1a-approved-execution-evidence/authorization.json`。
原SHA256：`6abee8f1a7ab7fbec4f9134eba616aea133d63b2556d1eda735642bc1bb39023`。原文件原文、键名、`user_approval_reference` 和 `user_message_id=null` **全部不改**。本轮逐字节复核该哈希。关联执行提交 `2cc22439587976d87550bf411e22f625d30508d0` 和归档提交 `5c14c347de2b183d59d7bd3f604ec45d0b602da6` 可从本地Git验证。

原凭据创建时间 `2026-10-26T04:38:06.336366+00:00` 是执行记录时间，不是外部批准消息的时间。用户提供的外部对话日期为2026年10月26日（UTC+8）；精确平台时间、消息ID、平台导出和专用分享链接均为 `NotAvailable`。不补造新的用户“批准了”消息，不用文件mtime、Git时间或本地记录时间替代平台消息时间。

原记录就“Codex接收到一条user执行指令”而言与本地角色一致，但对外部作者与原始授权链描述不足。本追加勘误补全三层来源；不是回写历史、撤回已发生执行或认证不可见的平台日志。完整转录见 [用户提供的来源记录](m8i1b_authorization_source_transcript.md)，原授权文字另在本轮JSON封存快照中保持原样。

实际一次只读启动和额度消耗不变。本勘误及外部转录均不构成新的CARLA启动、TM Pilot、采集、训练或控制授权。
