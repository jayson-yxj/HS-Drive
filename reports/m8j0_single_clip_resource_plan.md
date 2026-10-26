# M8-J0 单片段资源计划（未获执行授权）

检查时间：2026-10-26T06:23:26.478538+00:00；环境 Europe/Berlin。只制定计划，未建立数据解压目录。

唯一候选 `DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz`，官方压缩字节数 **280,176,944**（280.177 MB / 267.198 MiB）。
预期 SHA256 `b040f5993726dfb3a1dc1aaf5e2e4f93404a0124272d6f47f2469fca53ca25f2`。来源为[固定官方清单](https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/docs/bench2drive_mini_10.json)。它不是已下载文件的实测哈希。
当前数据盘可用 177,104,601,088 bytes；该瞬时值不是配额，也不能证明未来空间足够。解压大小和条目数均未知。

未来申请的硬预算：单一官方文件；压缩区320MiB（一个文件，不另留重复完整副本）；累计tar解压流最多2GiB；实际落盘一帧最多256MiB；元数据/日志64MiB；目录硬配额3GiB；至少4GiB可用才执行；内存1GiB；CPU单线程；下载墙钟10分钟，清单/一帧读取10分钟。任何上限触发即失败并留证，不能自动扩大。数字是拟议限制，未声称能覆盖整个压缩包。

拟议独立目录 `/root/gpufree-data/hs-drive-experiments/m8j1-single-clip-review/`，当前未创建。执行顺序：

1. 许可澄清且服务器官方API/HEAD可访问后，固定HF修订和唯一对象；拒绝其他文件、镜像或自动全库下载。重新核对官方size/SHA及适用条款，有差异即停。
2. 独立授权后才用校验证书的客户端下载到独占 `.partial`；限制320MiB；完成后严格比对280176944bytes和官方SHA，再在同目录原子重命名。保留来源和公共重定向证据，脱敏签名URL。不绕过认证、配额或条款。
3. 不运行 `tar -xf` / `extractall`。流式扫描tar，不执行文件；禁止绝对路径、`..`、重复/大小写冲突、软硬链接、设备/FIFO、稀疏条目和越界PAX路径；普通目录和普通文件白名单。计数上限50,000、单条目64MiB、累计解压流2GiB，超限退出；gzip尾部/CRC或tar截断异常也拒绝。
4. 先索引名称/大小和附带LICENSE/NOTICE；出现额外条款立即暂停后续像素读取。只选首个完整六摄+anno的相同文件序号，最多一次复读归档、总累计解压不超过2GiB。只释放该帧白名单成员，最多256MiB；其他帧不解压落盘。若顺序/预算不容许，保持Partial，不改选片段。嵌套json.gz单独限制解压16MiB、JSON大小/深度；图像每张不超过10M像素，拒绝解压炸弹。
5. CPU检查六摄分辨率、K主点/焦距、T逆及坐标轴、相机名称/朝向、GT角点投影与实例关系；保留源数值，不复制HS K/T，不改M8-B Adapter。缺原frame/timestamp则明确null；序号只能做离线索引。
6. RGB+审查标定走输入路径；Actor、Depth、Instance、Semantic仅走监督/诊断。TOP_DOWN、LiDAR、Radar、expert action均不进入正式RGB推理。无需CARLA、PETR、CUDA或训练依赖。

验收只覆盖一帧Schema与文件完整性；不证明同步时序、远场身份、完整数据质量、独立Final或训练资格。本轮未提供可自动执行下载/解压的命令。
