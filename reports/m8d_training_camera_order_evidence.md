# M8-D 相机顺序来源证据

固定原始 PETR 提交 f7525f93467a33707ef401c587a52d5e7b34de74；Checkpoint SHA256 16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad。

结论：**TrainingPKLOrderNotVerified**。可以验证官方代码的输入链，但没有取得这个 Checkpoint 实际训练所用的 info PKL。B 的正式名称是 **OfficialConverterOrderCandidate**，不能称作已确认的预训练正确顺序。

## 单帧 C5 代码链

1. [nuScenes converter](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/tools/data_converter/nuscenes_converter.py) 第197行 camera_types 为 FRONT、FRONT_RIGHT、FRONT_LEFT、BACK、BACK_LEFT、BACK_RIGHT，依次 update 到 info['cams']。保存 mmcv.dump 到 nuscenes_infos_train.pkl；Python 字典和 pickle 保留插入顺序。
2. [C5配置](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/projects/configs/petr/petr_r50dcn_gridmask_c5.py) 指向该文件名、CustomNuScenesDataset、LoadMultiViewImageFromFiles；没有 MultiSweeps loader。
3. [get_data_info](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py) 第52行遍历 info['cams'].items()，同一循环追加 filename、timestamp、intrinsics、extrinsics、lidar2img。
4. 实际安装的 mmdetection3d LoadMultiViewImageFromFiles 按 filename 列表读入再依序拆分；DefaultFormatBundle3D stack 保序；Collect3D 将 metadata 字段原样收集，不重排。
5. [ResizeCropFlipImage](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py) 以 i 遍历图像，对同索引 intrinsics 更新，再通过 intrinsics[i] @ extrinsics[i].T 构建 lidar2img。Normalize/Pad保序。
6. [SinePositionalEncoding3D](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/projects/mmdet3d_plugin/models/utils/positional_encoding.py) 在相机维 cumsum(1) 并归一化，编码camera index；几何标定相同不意味着相机序号编码相同。

实际服务器上完成了一个可信本地合成字典的 mmcv.dump/load 往返，并调用官方 CustomNuScenesDataset.get_data_info，验证 filename/K/T/projection 的顺序保持。此合成机制测试不是实际训练 PKL 审计。源码文件 SHA 和执行记录见 m8d_source_evidence.json / m8d_logs/source_audit.json。本地数据及仓库搜索未找到实际训练 PKL；未下载不明第三方 PKL，也未绕过数据访问条件。

## 其他官方脚本

- nuscenes_converter_seg.py 的 camera_types 与本轮 B 相同，另有分割监督流程，非当前单帧检测标签来源。
- generate_sweep_pkl.py 的 sensors 为 FRONT、FRONT_RIGHT、BACK_RIGHT、BACK、BACK_LEFT、FRONT_LEFT，与 B 不同。它从已有单帧 infos 读取 keyframe，并追加/替换 sweeps；不重建 keyframe cams。
- LoadMultiViewImageFromMultiSweepsFiles 默认 sensors 又显式指定 B 顺序，通过名字读取 sweep 字典，不依赖该字典插入顺序。它不在当前 C5 配置中。
- converter 中另一 camera_types 出现在2D COCO导出，不构成另一种当前网络输入对照。

本轮只有 A 与指定 B。没有使用 sweep 顺序作为第三组，没有参考高度、裁剪或相机排列搜索。以上证据支持比较官方转换器候选顺序，不足以追认实际训练资产。
