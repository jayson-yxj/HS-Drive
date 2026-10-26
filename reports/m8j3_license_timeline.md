# M8-J3 许可证据补充与范围更新

检查日期：2026-10-26。**LicenseConflictUnresolved**。沿用 M8-J2 已恢复的元数据访问结论，不重复网络排障。本记录是证据整理，不是法律意见。

| 时间与证据等级 | 官方事实 | 不支持的延伸结论 |
|---|---|---|
| 2024-08-14，J0 保存的 HF commit 页面日期；J2 固定字节已校验 | HF revision `ce56cfbc26edf7796586e26fd44644f0ac2f6bcc` 数据卡 YAML 与正文声明 Apache-2.0；J2 已将同一快照与目标 LFS 文件绑定 | 不证明该声明构成后来 GitHub 条款的资产级例外 |
| 2024-11-02 20:40:01 +08:00，官方 patch 作者时间 | `fd3009a16f551c6d36139c831a4a172f17b74419` 将 LICENSE 从 Apache-2.0 改为 CC BY-NC-ND 4.0 | 作者时间不等于经法律确认的生效时刻；不推断既有授权被撤销 |
| 2024-11-02 21:13:26 +08:00，官方 patch 作者时间 | `8f9f1da103ea7153a17c5b1dd046548a9bcd5af2` 将 README 的资产及代码许可说明改为 CC-BY-NC-ND；间隔 33 分 25 秒 | 不自行确定旧 HF 文件的适用范围或商业使用条件 |
| NeurIPS 2024 正式论文，精确上传/修订时间未建立 | 印刷第 25 页、PDF 第 25 页、I License 明确声明 Apache-2.0；本机 Ghostscript 从官方 PDF 指定页复核 | 不能凭会议年份断言 PDF 早于或晚于上述两次提交 |

论文短摘录：“All data, codes, and checkpoints are in GitHub and Huggingface under Apache License 2.0.”

官方链接：[论文](https://papers.nips.cc/paper/2024/file/017761f94a1cd66d01c041aff85492c4-Paper-Datasets_and_Benchmarks_Track.pdf)、[LICENSE 修改](https://github.com/Thinklab-SJTU/Bench2Drive/commit/fd3009a16f551c6d36139c831a4a172f17b74419)、[README 修改](https://github.com/Thinklab-SJTU/Bench2Drive/commit/8f9f1da103ea7153a17c5b1dd046548a9bcd5af2)、[固定 HF 卡](https://huggingface.co/datasets/rethinklab/Bench2Drive/blob/ce56cfbc26edf7796586e26fd44644f0ac2f6bcc/README.md)。

推断仅限于：不同渠道的声明可能未同步，或适用版本/资产范围不同。现有材料不能区分这些解释。维护者尚需明确目标文件是否受 GitHub 变更影响，内部训练、权重、衍生数据、商业使用的条件，以及第三方 NOTICE。

本轮后续用户明确确认个人、非商业用途，并授权下载指定公开片段和本地审查，不再以维护者澄清作为下载前置条件。我们依据该执行授权完成下载，**没有将用户用途声明写成维护者授权或冲突已解决**。不公开原始/处理数据、可视化或权重，不进行训练。

历史 J0/J1/J2 的 NoGo 和 PayloadNotVerified 保持原文；J3 对实际取得的唯一片段另记校验与有限 Schema 结果。未来商业使用、公开权重及衍生数据仍需单独明确适用条款。

补充文档仅 4 次固定官方 GET（两个 patch、论文、CARLA ContentVersions），均成功，TLS 校验开启。完整论文保留在本地私有目录；仓库只保存 URL、SHA 与短摘录。HF 卡复用 J2 原始字节。[证据附件](m8j3_evidence_attachment.md)。
