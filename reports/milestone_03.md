# M3 Dataset v1 工程验收

2026-10-19，实际服务器执行。**M3本次要求的标准化索引、按场景划分、PyTorch训练加载器和监督字段隔离已完成并通过工程门禁。** 保持HS-Drive V1.3架构。机器证据：[m3_gate.json](m3_gate.json)、[完整数据回放](m3_dataset/gate.json)。

## 数据与划分

派生数据位于 `/root/gpufree-data/hs-drive-data/processed/hsd-dataset-v1-m3`。版本ID为`hsd-dataset-v1-m3-smoke`，刻意表明这是工程验收集。

| 集合 | 场景 | 帧数 |
|---|---|---:|
| train | Town03 spawn0 正午静止、日落静止、低速运动；同一路线组 | 373 |
| val | 新采集Town03 spawn45，25m目标车辆 | 61 |
| test | 新采集Town10HD_Opt spawn0，低速运动；未见地图 | 61 |
| 总计 | 5段录制 | 495 |

新采集均为Epic、960×576、training-960-v4、逐tick渲染统一保留10Hz，61帧各6秒仿真，12路完整，无同步丢帧；日志`logs/m3_capture.log`。Town03训练与验证ego轨迹最近320.238m，超过100m预设门槛。所有原位置重复光照都留在train，不按帧随机拆分。三个集合的route group互不重叠，Town10仅在test。空间距离不是视觉内容完全独立的数学证明。

## 验收结果

- 495帧全部通过原始RGB/Depth/Actor文件校验和、尺寸、同帧传感器位姿、时间戳及标定检查。
- 2个spawn worker回放全部495帧；输入顺序、数值哈希及固定种子shuffle可复现。
- 单帧输入为[B,6,3,384,640]；因果三帧输入为[B,3,6,3,384,640]，train生成367个三帧窗口，不跨场景、不跨缺帧、不使用未来。
- Depth监督在原分辨率上最近像素采样，附准确UV，3×3深度突变边缘剔除；48个LID bins覆盖[1.5,80)m。有效监督比例train76.025%、val62.783%、test72.628%，不是感知准确率。
- RGB与K一起缩放；射线一致性单元测试通过。原始Depth不做面积插值，超界/NaN标签使用mask和ignore_index=-100。
- 推理路径不打开监督文件；删除所有targets/Depth/Actor/原始frames后仍成功。篡改Depth时训练失败但RGB输入不变。
- 输入额外键、白名单键中的嵌套GT、非张量、错误形状均拒绝；非法`actor_gt`访问抛异常并记录。这里只防工程误用，不宣称Python安全沙箱。
- CPU小型1×1 Conv接口探针的masked depth CE有限且反向梯度有限；**不是BEV网络训练或精度实验**。
- 数据回放与接口门禁13项全通过。仿真环境67项pytest通过，PyTorch模块因环境隔离跳过；数据环境71项通过、5项CARLA API测试跳过，后者已在仿真环境验证。两环境合计覆盖76个测试用例，均无失败。
- 两环境pip check通过；compileall通过。PyTorch为官方2.8.0+cpu，环境锁`requirements-data.lock.txt`，与`.venv`仿真环境分离。

可视化：[训练样本](m3_dataset/train_depth_targets.png)、[验证样本](m3_dataset/val_depth_targets.png)、[未见地图样本](m3_dataset/test_depth_targets.png)。彩色点为监督Depth采样射线，不是模型预测。

## 代码与命令

- `datasets/index.py`、`split.py`：不可覆盖、原子发布的派生索引、去重/路线组/空间约束。
- `datasets/training.py`、`inputs.py`：独立inputs/targets/metadata、因果窗口、批处理、受审计模型边界。
- `datasets/labels.py`：深度采样、有效掩码、LID/均匀分箱。
- `tools/build_dataset_index.py`、`validate_dataset.py`：实际构建和全量验收。
- `configs/dataset_v1.yaml`、`train_data.yaml`：来源清单与可直接加载的数据配置。

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
.venv-train/bin/python -m pytest -q
.venv-train/bin/python -m tools.validate_dataset \
  --index /root/gpufree-data/hs-drive-data/processed/hsd-dataset-v1-m3 \
  --workers 2 --output reports/m3_dataset_repeat
```

重建和代码示例见[数据接口与部署说明](../docs/M3_DATASET.md)。输出目录必须未存在，不能覆盖当前证据。

## 本次关闭边界

当前只有495帧、三个路线组、两张地图，足够验证数据工程，不足以验证驾驶泛化或正式模型质量。静态数据占比高，后续需扩大连续驾驶、交通参与者与天气覆盖。

自车状态来自CARLA理想snapshot，转换到局部episode起点与ego坐标，作为里程计代理；绝对世界位置不进入模型输入。导航指令没有采集，以−1明确标为未知。不得用该数据声称已实现导航规划。

已有Depth和Actor box监督；尚无drivable、Free/Unknown、Occupancy、未来轨迹标签。M4各损失启用前需实现对应标签。CPU DataLoader和反向探针不等于CUDA训练环境可用；CUDA wheel下载因网络慢未完成，下一步另建/验证GPU训练环境。

本轮结束时测试Actor归零、世界恢复异步、CARLA服务端已停止，GPU空闲。历史M2与成像门禁保持各自原始范围。
