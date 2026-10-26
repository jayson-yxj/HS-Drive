# M8-J2 实施与复核

基线 `5d6252a6f5a536d29cce5d2fb756134dec4057fe`；分支 `feat/m8j2-bench2drive-proxy-access-recheck`。只新增M8-J2文件，历史M8-J0/J1资产不改，不合并main。

访问范围固定为旧Mini单片段和官方GitHub/HF许可、元数据。`configs/m8j2_review_protocol.json`记录实际发现的代理端口、方法白名单、请求/超时/字节上限。新的元数据venv位于 `/root/gpufree-data/hs-drive-experiments/m8j2-metadata-env`，用system-site-packages复用系统requests；4个新包仅安装于该venv，记录见`reports/m8j2_environment.json`。无训练依赖安装或GPU任务。

实际运行过的命令（仅执行记录，**不要自动重跑联网阶段**）：

```bash
timeout 360 /root/gpufree-data/hs-drive-experiments/m8j2-metadata-env/bin/python -m tools.m8j2_access_probe
timeout 90 /root/gpufree-data/hs-drive-experiments/m8j2-metadata-env/bin/python -m tools.m8j2_complete_metadata
```

首轮源码保存在`reports/m8j2_sources/probe_executed_initial.py.txt`，其SHA对应http00–05。源码更新处理URL规范化后，http06–08绑定probe_executed_completion.py.txt的SHA；当前probe仅补齐错误文本IP脱敏边界，未重新联网，完成器本身也记录SHA。首轮本地AssertionError和所有失败保存；没有将首轮误写为全程成功。第二次resolver HEAD用于获取未持久化的临时URL，随后一次存储HEAD；只完成首轮未发送的有限检查。两次记录阶段合计仍受18请求预算限制。当前所有网络阶段已结束，未用额度不构成新增实验许可。

每个Python请求保留UTC、客户端/代理模式、TLS、HTTP状态和公开头。实际存储URL的签名参数只在进程内短暂使用，归档为REDACTED。Git原trace包含潜在敏感头，因此只保存trace SHA、CONNECT/TLS/方法证据，不保存原trace。GitHub错误体带出口IP，已脱敏并保留原响应SHA，不作为许可证据。本机DNS记录的是目标解析，不是私人出口。没有检查或发送HF凭据，HF客户端使用token=False。

官方身份由固定Mini清单、两种Python API、Git HEAD、resolver元数据和storage HEAD互证。存储ETag/Xet与LFS归档SHA区分；没有实际归档字节供完整性验证。API表示公开/非gated仅是访问条件，不是使用授权。

可直接执行的离线复核：

```bash
python3 -m unittest tests.test_m8j2_access -v
python3 -m tools.m8j1_finalize verify
python3 -m tools.m8j2_finalize verify
git diff 5d6252a6f5a536d29cce5d2fb756134dec4057fe --name-status
```

`tools/m8j2_finalize build`从已有证据生成新报告，采用独占创建，不覆盖已有交付。`reports/m8j2_release_manifest.json`绑定源码、配置、报告和测试记录的SHA，提交由Git记录提供。合成测试验证请求边界和脱敏；离线校验验证资产/报告一致性，均不认证数据内容、模型、训练或驾驶安全。

决策为B，访问恢复但许可未明；维护者询问未发送。[总结](../reports/m8j2_summary.md)、[数据源决策](../reports/m8j2_source_decision.md)、[验收](../reports/m8j2_acceptance.json)。
