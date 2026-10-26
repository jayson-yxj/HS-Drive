# M8-J0 交付前离线核验勘误

2026-10-26。首次release验证未通过，原因是新建`github_utils.decoded.txt`由文本帮助函数附加了一个换行，实际7857bytes与API解码原文7856bytes不同。失败发生在新副本校验，M8-I0及M8-I1b原文未改。已按保存的API base64精确恢复原字节，最终生成器改为二进制写入，未经重新联网。

实际运行的生成器保存在`m8j0_sources/finalize_executed.py.txt`，其SHA对应报告`generator_sha256`；实际探针两版同样保存。最终源文件包含边界防护和字节保真修正，不把修正后工具说成已经重新访问网络。

首次清单另存`m8j0_sources/release_manifest_initial_failed.json`。该清单记录的是修正前交付快照，不作为当前验收清单。最终以`m8j0_release_manifest.json`和离线验证器结果为准。实施文档命令块也已重建为独立行。

原请求日志UTC为真实采集时间；首六条当地显示用UTC+8，最后一条以及汇总使用Europe/Berlin。本轮没有改写原始网络请求记录。离线勘误不消耗或产生CARLA/下载权限。
