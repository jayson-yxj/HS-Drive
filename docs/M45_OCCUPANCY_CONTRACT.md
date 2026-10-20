# M4.5 Observed Occupancy 合同 v1

版本 hsd-m45-observed-occupancy-v1；入口 tools.build_occupancy_oracle。标签是独立 sidecar，未加入旧数据索引，也未加入模型输入。datasets.schema.validate_learned_inputs 继续以 allowlist 阻止监督字段进入纯视觉推理。

## 坐标和含义

所有数组 [Z,X,Y]=[4,160,120]，HS ego：X 前、Y 左、Z 上。X=[-20,60)、Y=[-30,30)，XY=0.5m；Z 边界 [-2,-0.5,1,2.5,4]，所有上边界排除。不得转置为图像行列后写回标签。

| 字段 | 类型 | 定义 |
|---|---|---|
| occupancy_state | uint8 | 0 UNKNOWN，1 FREE，2 SURFACE_OCCUPIED |
| occupancy_valid | bool | state != UNKNOWN；只有此 Mask 内允许计算 observed occupancy loss |
| surface_semantic | uint8 | 0 UNKNOWN，1 GROUND，2 STATIC_OBSTACLE，3 DYNAMIC_OBSTACLE，4 OTHER |
| surface_semantic_valid | bool | 有无冲突的已知表面语义 |
| obstacle_mask / obstacle_valid | bool | static/dynamic 为正，ground 为负；unknown/other/冲突不是负样本 |
| visibility_count | uint32 | 接受射线的 free 穿越或 endpoint 观测次数，不是置信度 |
| surface_count | uint32 | 接受的表面 endpoint 数；自车粗体素屏蔽前的诊断计数 |
| ray_conflict_mask | bool | free 路径遇到其他已知 surface，已截断 |
| semantic_conflict_mask | bool | 同体素 ground/obstacle 等语义冲突；语义和障碍监督无效 |
| ego_mask | bool | 与加裕量自车包围盒相交的体素，occupancy 强制 UNKNOWN |

FREE 表示可靠射线观测到穿过该体素的自由空间证据，并不证明粗体素内部每一点都为空。SURFACE_OCCUPIED 表示观测表面命中，不是物体内部填充；地面也属于表面，不能用此状态直接当作不可驾驶障碍物。UNKNOWN 不进入负类监督。Drivable、道路规则及通行权维持原独立合同。

## 射线与冲突

1. 使用原 RGB/depth 标定：相机真实 optical 原点；CARLA optical-Z 反投影得到 endpoint，射线长度由 endpoint-origin 求得，不能把 Z 深度直接当欧氏距离。
2. 固定 stride=16；有效 optical-Z 范围 [1.5,80)m；原生 3×3 patch 全部有限且极差 <=0.5m。深度无效、截断或边缘不可靠射线不标 Free。
3. FP64 枚举射线与所有 X/Y/Z 网格面的交点，以相邻交点中点确定正长度穿越体素。相机可以在网格外；处理非均匀 Z 层，无固定大步长跳格。
4. 射线 free 终点从实际 surface 回退 0.1m；endpoint 本身仍单独标表面。这是沿射线距离裕量，不是膨胀 occupied 体素。
5. 先记录所有可靠 endpoint，随后 free traversal 到其他已知 occupied 即停止，保留 occupied 并标冲突；其后不继续 carve Free。独立射线的可靠远端 surface 仍可保留，并非将遮挡后整段填满。未解决的语义冲突置语义 Unknown，可靠的几何表面事实保留。
6. 自车 local bounding box 增加0.05m裕量。任何穿过此包围盒的射线整条丢弃；相交粗体素为 Unknown。这是保守近似，包围盒可能覆盖实际空气。当前清晰样本丢弃2543/4098条有效深度射线，地面覆盖很稀疏；在审查实例 self-mask/车体 mesh 前不得悄悄解除过滤。

## 语义真值

新录制同步采集六路 RGB、Depth、Semantic，frame/timestamp 对齐。运行时读取 CARLA 0.9.15 的 CityObjectLabel 枚举并保存映射，避免用旧版本数字表。Road/RoadLine/Sidewalk/Ground/Terrain/RailTrack 合为 GROUND，建筑/墙等为静态类，Car/Pedestrian 等为可运动类。DYNAMIC_OBSTACLE 不代表此帧正在运动。Sidewalk 属物理 ground，不表示允许通行。Sky/Water/未识别保持 unknown，旧录制没有语义时不制造语义负样本。

## 实测与门禁

9帧（Town04旧驾驶3，Town03 clear/fog各3）完成生成；实际覆盖约15%–20%，未观测仍 Unknown。合成墙后 Unknown、多相机、半开边界、ground/obstacle、self-mask、无效深度与冲突测试通过。详细每帧/每层计数见 reports/m45_occupancy_validation.json；示例 NPZ、四层图、source manifest 见同名前缀 figures。

工程标签生成可用；通用 Occupancy 训练数据质量 Partial：语义只覆盖同一静态位置，地面极稀疏，self-box 过保守，四层高度会混合地面/障碍，未审查真实 mesh 可见空间误差。可以做 masked 数据管线 smoke test，但本轮没有训练 Occupancy Head。先完善 self-mask 和多个道路/遮挡/坡道场景的标签，再做小规模训练；不将9帧的稳定生成当成泛化证据。
