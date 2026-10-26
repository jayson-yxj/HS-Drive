# M8-J2 — 代理恢复后的Bench2Drive访问复核

检查日期：2026-10-26。基线 `5d6252a6f5a536d29cce5d2fb756134dec4057fe`；独立分支 `feat/m8j2-bench2drive-proxy-access-recheck`。

**最终决策 B：`AccessResolvedLicenseUnresolved`。** 显式代理下官方元数据和目标存储HEAD访问恢复，许可仍有实质歧义。`TrainingReadinessNoGo / PayloadNotVerified`保持。

服务器实测发现运行中的verge-mihomo，配置和监听一致为loopback混合端口7897、rule模式、TUN关闭。当前shell八个大小写代理/NO_PROXY变量均未设置，Git没有显式代理配置。本轮仅在独立诊断进程使用 `http://127.0.0.1:7897`，没有修改全局配置。

| 客户端/层级 | 实测 |
|---|---|
| DNS | 保存本机实际解析结果；HTTP CONNECT将目标解析交给代理，未认证代理上游DNS或出口。 |
| requests默认路径 | ConnectionError / network unreachable；未建立HTTP响应。 |
| requests显式代理 | 仓库API200，socket对端为loopback7897，TLSv1.3且证书验证开启。 |
| huggingface_hub0.26.5 | 通过同一显式代理backend，固定revision API200、目标HEAD302，无下载调用。 |
| Git | 单次ls-remote HEAD成功，trace证实loopback7897/CONNECT/证书验证/HTTP200；无clone或LFS。 |
| 官方HF存储 | 仅跟随HF签发的存储URL一次HEAD，200，application/gzip，Content-Length280176944；载荷正文0bytes。 |
| GitHub新请求 | 两次403，正文为匿名API限流；未绕过或重试，不解释为许可拒绝。 |

官方目标身份已闭合至元数据层：

- 仓库 `rethinklab/Bench2Drive`，`private=false / gated=false`，匿名读取。
- revision/Git HEAD：`ce56cfbc26edf7796586e26fd44644f0ac2f6bcc`。
- 唯一文件 `DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz`，280176944bytes。
- LFS SHA256：`b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2`，与固定官方Mini清单一致。
- LFS pointer Git blob `94021e690ca19263ae880aa5f371f1f64deac66b`；Xet hash `0415f4b9744680509bd5020372502a034b5992923fb0b704bd2daca072446f70`。两者不冒充归档SHA。

固定revision的HF README本机HTTP200，仍声明Apache-2.0；GitHub固定CC BY-NC-ND原文沿用已核验官方证据、重新检查SHA。没有解决两者的资产适用差异，也没有读取包内第三方条款。HTTP200不等于训练授权。[许可对照与未发送补充询问](m8j2_license_comparison.md)。

本轮有9条Python HTTP记录、1条Git引用广告请求；3次目标/存储HEAD，目标GET与Range均0。首轮采集器在HEAD302后因签名URL白名单匹配触发本地AssertionError，发生于存储请求前，失败记录保留。完成一次有限修正后重发resolver HEAD取得临时URL，再进行唯一存储HEAD200；没有自动重试循环或GitHub限流重试。总请求仍在预声明18次上限内。签名query、凭据及GitHub限流正文中的出口IP均不进入公开报告。[原始证据及边界](m8j2_network_proxy_audit.json)。

因现有环境缺少huggingface_hub，本轮新建仅元数据用途的独立venv，并从官方PyPI安装4个小依赖；没有改变项目.venv或历史GPU环境，没有下载目标数据。依赖安装与数据访问计数分开，版本及安装RECORD SHA见[环境记录](m8j2_environment.json)。

73个M8-J0/J1交付文件冻结核验，原306项历史资产检查通过；7项合成请求边界单测通过。旧报告、默认模型和控制未改，没有CARLA、解压、训练或推理。维护者消息发送0。

后续保持许可No-Go；不自动下载。具体访问证据、资产身份、决策和复核命令：[网络](m8j2_network_proxy_audit.json) · [身份](m8j2_official_asset_identity.json) · [验收](m8j2_acceptance.json) · [决策](m8j2_source_decision.md) · [实施记录](../docs/M8J2_IMPLEMENTATION.md)。
