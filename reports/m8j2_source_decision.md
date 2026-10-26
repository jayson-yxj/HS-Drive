# M8-J2 数据源决策

2026-10-26。最终选择 **B：`AccessResolvedLicenseUnresolved`**。

访问在**明确配置代理的本轮进程内**恢复：Python requests、huggingface_hub和Git均取得官方数据仓库元数据；固定目标经官方resolver302到HF存储端点，HEAD200返回application/gzip及280176944bytes。LFS SHA与冻结清单一致，Xet标识另行记录。完整文件传输、实际内容和归档内条款均未验证。

默认shell的HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/NO_PROXY及小写版本仍全部未设置；代理进程存在但TUN配置关闭。默认requests路径失败，显式HTTP CONNECT路径成功。这支持当前访问差异与代理配置相关，但不能追溯证明M8-J0/J1每次失败唯一由同一原因造成。没有更改全局shell、Git、DNS、路由或历史环境。

不选A：固定HF数据卡与固定GitHub声明的资产级适用关系仍未澄清，未得到明确第三方范围。HTTP200和匹配SHA不能提供训练授权。

不选C：官方匿名仓库、精确目标身份及存储HEAD已有成功服务器证据。默认shell未配置代理和GitHub独立API限流是已明确的访问限制，不否定已证明的HF元数据通路。

不选D：目标确实位于官方快照，公开且未gated，元数据匹配；没有明确不适用或禁止用途的资产说明。

后续建议：保留这条已验证的显式代理元数据路线；优先获取官方针对该片段的许可澄清。维护者询问未发送，如需发送须独立授权。许可未闭合之前不提出下载执行申请，不扩大数据源/下载器工程，不训练或运行模型。若未来另行批准读取，还需完整文件SHA、安全单帧审核、原六摄K/T、时间缺口及静态/实例身份审查。官方Train不等于HS新Final，单帧Schema不等于训练质量或驾驶安全。

`TrainingReadinessNoGo / PayloadNotVerified`保留。原M8-J0/J1报告和失败记录均不改写。本轮不启动CARLA，不解压数据，不模型前向，不合并main。

依据：[网络证据](m8j2_network_proxy_audit.json)、[资产身份](m8j2_official_asset_identity.json)、[许可对照](m8j2_license_comparison.md)、[验收](m8j2_acceptance.json)。
