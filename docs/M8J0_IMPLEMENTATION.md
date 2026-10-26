# M8-J0 实施及复核

基线 `fe8db837d18f033e446475d0bf48534eb54bc0f3`；分支 `feat/m8j0-bench2drive-license-access-review`。2026-10-26在HS工作区服务器执行。只新增m8j0/M8J0文件，历史结果、源码及默认行为不改；不合并main。

`configs/m8j0_review_protocol.json`冻结唯一文件、官方SHA、请求预算、提案条件。`tools/m8j0_probe.py`只允许白名单元数据GET与唯一文件HEAD；HTTPS证书验证保持开启，无Range、自动重定向和重试。每请求25秒、最多18次，总公开正文6MiB、单元数据响应2MiB。实际7次、39,060bytes。未安装下载、训练或CARLA依赖。

来源、证据与复现边界：

- `reports/m8j0_sources/*.request.json`为实际本机请求，保留时间、URL、方法、HTTP类别、curl退出码、公开头和错误信息；`.body`仅为GitHub元数据。
- `probe_executed.py.txt`与`probe_executed_utils.py.txt`保存实际执行的两版探针源码，与原记录source SHA对应；最终工具另有静态防护加固，没有重新联网执行。旧HEAD命令带max-filesize，连接在响应头之前失败，不能据此测量大对象；最终HEAD已去除该正文限制。
- 首六条原始日志当地时间显式为UTC+8，与本会话Europe/Berlin不同；UTC时间不变。汇总另算Europe/Berlin显示，未回写原始日志。
- `browser_*capture.json`是浏览研究服务官方页面摘取，可能缓存；其网页读取成功不充当服务器HTTP成功，工具错误也不冒充服务器状态码。完整访问范围见browser研究记录。
- 官方固定源码复用M8-I0冻结原文，并重新核验SHA256和Git blob。官方Mini清单本轮API返回与旧字节完全一致。新增utils/CARLA README只用于源码/许可审查，没有执行这些官方模块。
- `tools/m8j0_finalize.py`仅从证据生成JSON和资源计划，AST读取相机常量，不导入CARLA；已存在输出用`open('x')`拒绝覆盖。它不应在当前交付目录重复运行。准确复核使用下面的verify命令。

可直接执行的**离线**验证：

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
python3 -m unittest tests.test_m8j0_review -v
python3 -m tools.m8i1b_verify
python3 -m tools.m8j0_verify
git diff fe8db837d18f033e446475d0bf48534eb54bc0f3 --name-status
```

验收器还检查相对于准确基线未修改/删除任何旧文件。源代码、JSON、日志、许可原文引用及所有新增文件SHA由`reports/m8j0_release_manifest.json`绑定；提交SHA以Git分支记录为准，避免在自身文件中填写自指哈希。

本轮实际网络命令（仅复现记录；已结束，不要求再次执行）：

```bash
python3 -m tools.m8j0_probe github_manifest GET 'https://api.github.com/repos/Thinklab-SJTU/Bench2Drive/contents/docs/bench2drive_mini_10.json?ref=7ec25d1c9f7522d923ce5f3420986cef1cb2d956'
python3 -m tools.m8j0_probe hf_repo GET 'https://huggingface.co/api/datasets/rethinklab/Bench2Drive?blobs=true'
python3 -m tools.m8j0_probe hf_asset_head HEAD 'https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main/DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz'
```

其余4次GitHub GET URL在逐请求记录，分别为许可历史、仓库内license问题搜索、固定CARLA README及固定B2D utils。初始冻结辅助命令因Python3.10没有hashlib.file_digest失败且未写输出；改为流式SHA完成，306项原证据及补充清单再次通过。两次HF失败后未扩大网络试探。

测试层级：8项合成请求边界测试；官方清单/源码Git blob和369路径记录（357实际文件）离线核验；7次真实网络元数据检查。CARLA、完整下载、解压、标定载荷审核、模型、训练和驾驶全部NotRun。任何一个离线检查通过不提升这些门禁。

完整结论见[验收](../reports/m8j0_acceptance.json)、[总结](../reports/m8j0_summary.md)、[架构决策](M8J0_ARCHITECTURE_DECISION.md)。
