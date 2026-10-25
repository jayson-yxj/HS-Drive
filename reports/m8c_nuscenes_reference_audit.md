# M8-C nuScenes / PETR 参考分布审计

本轮固定PETR源码 `f7525f93467a33707ef401c587a52d5e7b34de74`、检测权重SHA `16d8a4875c36e7d1cba75b357901ff031b18d9d9374f12780b9b72cc621cefad`。本地源码逐文件SHA见camera_distribution_comparison。网络获取的官方资料完整保存于独立实验目录official，URL/HTTP结果/字节数/SHA见m8c_logs/official_access*.json。未安装新依赖。

## 可验证的代码合同

- [官方数据转换器](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/tools/data_converter/nuscenes_converter.py) 读取当前sample的LIDAR_TOP，保存lidar2ego旋转/平移；camera sensor2top还组合各采样时刻ego-to-global变换，并非只使用静态rig。GT中心由nuScenes `get_sample_data(lidar_token)`提供，位于该LiDAR参考坐标；速度从global旋转至LiDAR。GT尺寸wlh，yaw转SECOND格式 `-yaw-π/2`。
- [dataset](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py) 将sensor2lidar逆变换组合相机内参为lidar2img。名称不表示消耗点云；当前模型配置camera-only。HS当前使用ego actor origin为pseudo-LiDAR，数学投影正确不代表参考分布相同。
- [PETR Head](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/projects/mmdet3d_plugin/models/dense_heads/petr_head.py) 先在16×44特征格构造图像坐标，再以64个LID深度位置反投影。深度 `1+(61.2-1)/(64×65)×i×(i+1)`，i=0…63，最大59.3477m。这是PETR位置编码采样，**不是HS原48个LID Depth Bin**；二者均未修改。
- position_range固定[-61.2,-61.2,-10,61.2,61.2,10]；pc_range固定[-51.2,-51.2,-5,51.2,51.2,3]。归一化3D位置经过inverse_sigmoid夹到边界。computed geometry mask统计的是D×3坐标分量越界数量大于D/2；forward取返回值时丢弃该mask，Transformer仍用pad mask，不能声称越界几何自动屏蔽。
- 900个可学习reference points进入query embedding，六层回归逐层输出中心；中心XYZ用pc_range恢复，decoder最终全类别sigmoid分数展平top300，可能同一query以不同类重复出现，无NMS。post_center_range固定[-61.2,-61.2,-10,61.2,61.2,10]。Z输入输出中心与历史底面约定见M8-B；没有修改权重或范围。

## 新发现：相机序号是输入分布的一部分

官方转换器顺序：front、front_right、front_left、rear、rear_left、rear_right。
HS顺序：front、front_left、front_right、rear_left、rear_right、rear。

[SinePositionalEncoding3D](https://github.com/megvii-research/PETR/blob/f7525f93467a33707ef401c587a52d5e7b34de74/projects/mmdet3d_plugin/models/utils/positional_encoding.py) 对相机维做cumsum并归一化，加入相机index的sin/cos，随后adapt_pos3d进入Transformer。因此K/T跟随图片保证物理投影正确，**不保证预训练相机index语义一致**。官方模块离线数值检查，除front外五个相机重映射后编码最大通道差0.866–1.550。这里依据固定官方转换器推断原训练次序；原训练PKL未取得，不能把推断说成逐条核实。

本轮未重排模型输入、未用输出质量决定顺序。该发现支持 `ReferenceDistributionMismatchSupported_CameraIndexEncoding`，但未证明远距错误主要由其导致。应先申请独立单变量检查，不能与高度、裁剪同时更改。

## 官方资料实际可得范围

获取了[官方教程](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html)和固定devkit提交 `b40adc467b919192899405d9b77871afee8efa07` 的[Schema](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/docs/schema_nuscenes.md)、[数据访问说明](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/README.md)。教程给出真实sample/CAM_FRONT token及1600×900尺寸，标定示例token `f4d2a6c281f34a7eb8bb033d82321f79` 的sensor_token `47fcd48f71d75e0da5c8c1704a9bfe0a` 可在同页sensor表关联为 **RADAR_FRONT**。平移[3.412,0,0.5]绝非已证实的LIDAR_TOP高度；未用于B组。

本地未找到完整nuScenes calibrated_sensor/sensor/sample_annotation表；官方devkit树没有这些数据库文件。下载说明要求账号及条款流程，当前无已授权数据资产；没有采用第三方散落数值、绕过访问流程或猜测安装高度。最初历史代理端口拒绝连接，直接HTTPS读取官方资料成功，证书校验未关闭。原始LIDAR_TOP和六相机实测K/T、物理高度/基线/FOV、典型目标LiDAR中心Z/尺寸分布、可定量比较的nuScenes原始RGB样本均 **NotAvailable**。教程单例truck不作为训练目标分布。

因此物理B组 **ReferenceVariantNotRun**。只有synthetic transform合同测过，不把任意合成平移当真实标定。

## HS实际分布

192帧记录相机K/T；六摄fx=fy≈268.512px、源640×384、水平FOV约100°、垂直FOV约71.13°。前摄有约5°俯仰，安装z1.65m，其余z1.60m。基线距离矩阵、方向和完整T在JSON。官方增强2.2倍后垂直保留约44.23°；分辨率升采样不增加源信息。原155个GT几何中心Z median0.744m、range[-0.438,1.965]m。没有nuScenes同口径实测统计，不能断言差了某个固定高度。

实测位置采样any-XYZ越界每摄约23.2%–27.0%，主要为Z；近场0%、10–20m约6.44%、20–40m约44.74%、40m以后约73.52%。computed mask每摄1.99%–19.32%，但未被官方forward采用。越界量是几何采样，**不是目标概率质量**，亦不是深度预测的已证实错误。12帧同rig观测并不构成12个独立标定分布。
