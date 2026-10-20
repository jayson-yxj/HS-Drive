# M3 标准化数据与训练边界

继承V1.3网络架构、hsd-frames-v1录制格式及已验收training-960-v4成像参数。M3增加派生索引和PyTorch数据接口，不训练BEV模型。历史M2/成像报告不回写。

## 索引与划分

`configs/dataset_v1.yaml`显式列出允许导入的场景、跨录制路线组、种子和保留地图。只有完整录制、六摄、已知成像profile、完整标定/版本/种子/天气可导入；索引构建时验证每帧图像、Depth、Actor标签校验和、尺寸、时间戳、传感器位姿。失败时不发布半成品索引，已有输出不可覆盖。

相机实验的多个原始目录均叫`training-960-v4`，不能把目录名直接当全局scene_id。注册表为各录制指定唯一标准scene_id，同时保留原始scene_id、路径、标定ID、manifest/frame SHA256和source_sha256。

- 连续录制整体归属train/val/test，禁止按帧随机划分。
- 同一路段重复录制/不同光照必须设置同一group_id。显式split优先；未指定时使用固定seed和group_id的SHA256做稳定分组，跟输入顺序无关。
- Town10HD_Opt仅用于test；Town03独立位置用于val。
- 同地图跨集合检查所有采样ego XY位置最小距离，当前阈值100m；跨集合完全相同的六摄RGB校验和组合也被拒绝。
- 空train/val/test会报错，不自动把相邻帧填到缺失集合。

距离阈值和显式路线组防止明显重复，但不能证明所有远景、路口或纹理完全独立。扩展路线须人工维护group_id；不把本次小型工程集称为模型效果基准。

结构：

```text
processed/<version>/
  manifest.json             # 最后提交标记、index SHA256
  index.json                # 场景、来源、划分、样本引用
  splits/{train,val,test}.json
  inputs/<scene>/<frame>.json   # 仅模型合规字段与RGB引用
  targets/<scene>/<frame>.json  # Depth/Actor引用，独立打开
```

RGB/Depth大文件仍在raw_root，索引不复制图像。迁移数据盘可在HSDriveDataset中传`raw_root=`，相对路径及校验和保持有效。索引一次加载元数据；图像逐样本解码，不缓存整段视频。

## 张量接口

`HSDriveDataset`返回`DataItem(inputs, targets, metadata)`，`collate_items`返回同结构批次。**模型只能接收`batch.inputs.as_model_kwargs()`，不能接收DataItem或targets。**

| 字段 | 单帧批次形状/类型 | 语义 |
|---|---|---|
| images | float32[B,6,3,384,640] | 固定相机顺序RGB /255；不做mean/std归一化 |
| intrinsics | float32[B,6,3,3] | resize后的K，OpenCV半像素中心规则 |
| cam_to_ego | float32[B,6,4,4] | hs_ego←optical |
| ego_pose | float64[B,4,4] | 片段起点ego坐标系←当前ego，无绝对地图位置 |
| ego_state | float32[B,7] | speed、ego-frame vx/vy/vz、ego-frame wx/wy/wz |
| timestamps_s | float64[B] | 从片段起点起的秒数 |
| route_command | int64[B] | 当前采集没有导航标注，−1表示未知 |
| valid_cameras | bool[B,6] | 已验收完整六摄均为true，不伪造丢失相机 |

顺序固定为front/front_left/front_right/rear_left/rear_right/rear。三帧模式为上述字段加入T维，例如images=[B,T,6,3,H,W]；窗口末帧为当前帧，只用历史，不补未来、不跨场景、不跨时间缺口，前两帧不产生三帧样本。`sequence_stride`是历史抽帧步长，仍检查整个覆盖区间的连续性。

自车位姿/速度来自同帧CARLA snapshot，作为允许里程计的**理想仿真替代**；不是实测轮速/IMU或视觉定位。metadata明确标记来源。未来应采集里程计噪声与定位失效场景。场景ID、地图/天气、原始绝对位置、Actor、Depth和标签来源不进入模型kwargs。

`LearnedInputs`只保存固定字段张量：额外键、嵌套字典、非张量和错误形状会被拒绝，访问`depth/actor_gt/...`触发KeyError并记录`hsdrive.input_audit`日志。这是工程接口隔离，不是对任意恶意Python代码的沙箱；无法判断有人故意把GT数值编码进RGB张量。

## Depth监督

targets提供depth_m/depth_valid/depth_bin=[B,6,24,40]、depth_uv=[B,6,24,40,2]；可选objects保持变长列表，包含原始is_ego标记。所有targets只进入损失/评测。

- 使用独立实验确定的optical_z，保留旧calibration内unverified字段，以索引的depth_evidence建立解释，不回写原始标定。
- RGB以面积插值缩小，K同步半像素变换；Depth不面积平均。
- 每stride16特征单元以`(j+0.5)*16-0.5`定义中心，映射回原图选最近原生像素。depth_uv记录该原生像素缩放后的准确射线位置；M4投影/损失必须遵循同一约定。
- 3×3原生邻域深度差>0.5m时屏蔽，避免遮挡边界标签。有效范围[1.5,80)m，非有限/超界无效；无效depth_m=0、bin=-100、valid=false，必须使用mask。
- 48个LID区间：edge_i=1.5+(80−1.5)i(i+1)/(48×49)，右端80m不属于最后Bin。函数另支持均匀区间用于后续消融。

最近采样、0.5m边缘阈值、LID均为明确可复现实验选择，并非已证明最优。当前没有drivable、Free/Unknown、Occupancy或未来轨迹标签，不把点数BEV冒充这些监督。

## 执行

```bash
cd /root/gpufree-data/hs-drive/HS-Drive_Server_Handoff_v1.0
bash scripts/bootstrap_data.sh
.venv-train/bin/python -m tools.build_dataset_index \
  --registry configs/dataset_v1.yaml \
  --output /root/gpufree-data/hs-drive-data/processed/hsd-dataset-v1-m3-repeat
.venv-train/bin/python -m tools.validate_dataset \
  --index /root/gpufree-data/hs-drive-data/processed/hsd-dataset-v1-m3-repeat \
  --workers 2 --output reports/m3_repeat
.venv-train/bin/python -m pytest -q
```

也可通过 `datasets.training.loader_from_config(index_dir, split, 'configs/train_data.yaml')` 调用已版本化的配置；val/test强制顺序读取，不继承train的shuffle。

加载示例：

```python
from datasets.training import HSDriveDataset, make_dataloader

def main():
    ds = HSDriveDataset('/path/to/index', 'train', supervision=True, sequence_length=1)
    for batch in make_dataloader(ds, batch_size=1, workers=2, seed=7, shuffle=True):
        model_kwargs = batch.inputs.as_model_kwargs()
        # outputs = model(**model_kwargs)   # M4实现网络；M3不提供假模型
        # loss = criterion(outputs, batch.targets)
        break

if __name__ == '__main__':
    main()  # spawn多进程需__main__入口
```

推理用`supervision=False`；不会打开targets、Depth NPZ或Actor JSON，原始混合frames JSON也无需读取。删掉监督文件后训练报错、推理仍可运行。每次加载默认校验实际读到的文件，明确拒绝陈旧索引/被篡改RGB/标签。

M3采用独立CPU PyTorch 2.8.0环境完成数据验收，选择来自[官方版本安装表](https://pytorch.org/get-started/previous-versions/#v280)。复用[PyTorch Dataset/DataLoader](https://docs.pytorch.org/docs/2.8/data.html)的spawn workers、Generator和worker_init_fn；默认batch1、prefetch1。官方CUDA wheel下载慢，本轮没有完成CUDA环境安装，也不宣称GPU训练可用。M4开始前须另建并验证CUDA环境，保持仿真.venv不变。
