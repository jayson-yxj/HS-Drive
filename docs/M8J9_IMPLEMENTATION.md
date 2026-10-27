# M8-J9 冻结A官方评测接口

基础136023d；独立分支feat/m8j9-frozen-a-official-eval。只读取81帧A缓存和本地nuScenes metadata；不加载权重，不构建模型，不新增前向或更新。B不进入本轮执行路径。

## 冻结合同

configs/m8j9_protocol.json冻结原始A标识、缓存清单、10类顺序、devkit1.1.9、detection_cvpr_2019、字段来源和源码SHA。保持原Top300、score≥0.2、strict XY±51.2m；原导出器应用class ego radius过滤。不是用低分缓存重新优化AP。官方每sample上限500，无需新增截断或NMS。

PETR CustomNuScenesDataset继承的_format_bbox来自冻结MMDetection3D0.17.1；直接调用，未修改模型、解码器、属性规则或评测算法。接口薄层只补齐已经存在的data_infos位姿字段和缓存tensor类型。它只读sample/sample_data/calibrated_sensor/ego_pose/scene/sensor，不构造会加载GT的NuScenes。缓存NPZ只访问boxes/scores/labels/query_indices/raw_bbox键，不访问gt/gt_labels。

坐标为reference LIDAR_TOP自己的ego pose，非HS Ego直接假设。底面z转换gravity center，完整wlh，官方yaw逆换为-yaw-pi/2，Quaternion顺序wxyz；先lidar→ego再ego→global。速度直接来自预测7:9，并与缓存raw8:10精确核对；原导出规则只补平面vz=0后按两个旋转变换，不从GT导入速度。属性由原导出器类别默认值和预测全局XY速度是否>0.2m/s确定，属于确定性规则，不是学习到的属性。

每一导出框用独立矩阵/四元数链校验center/size/rotation/velocity和score，容差1e-5（FP32源yaw/center允许舍入）。验证sample集合严格等于本地mini_val全体81帧，缺失不填空；真实零框样本允许保留空列表。合成不对称尺寸、中心、旋转、速度、属性、空预测、缺字段、GT隔离和token检查16项通过。

## 官方执行

导出、eval1、eval2为独立CPU进程，禁网、CUDA不可见、每阶段≤600s、失败停止、不自动重试。使用相同版本官方DetectionEval.main(plot_examples=0, render_curves=False)，同一私有results_nusc.json评测两次。NuScenes加载原始mini_val GT，官方filter_eval_boxes执行类别距离、零点数、bike-rack规则；这些GT信息只进入评价，不参与预测导出。

指标原始JSON原样保存，不修改官方类别汇总、匹配、积分、TP误差或NDS。部分官方NA字段为NaN，保留原格式并解释，不替换为0。重复比较仅剔除eval_time，其他全部结构和值（包括对应NaN位置）要求一致。

GT审计另行调用同版本官方load_gt/add_center_dist/filter_eval_boxes，核对结果与引擎对象一致，记录原始/过滤后数量、类别支持、instance数量及和旧3559GT集合差异。逐annotation比较文件仅私有保存。原有Z/高度/尾部诊断仍保留，官方指标不替代。

## 命令

```bash
M8J9_RT=/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  LD_LIBRARY_PATH="$M8J9_RT/env/lib/python3.8/site-packages/torch/lib:$M8J9_RT/cuda-11.8/lib" \
  PYTHONPATH="$PWD:$M8J9_RT/sources/PETR:$M8J9_RT/sources/mmdetection3d" \
  "$M8J9_RT/env/bin/python" -m pytest --noconftest -q tests/test_m8j9_official_export.py

# clean committed checkout; creates new isolated private outputs, refuses existing directories
.venv/bin/python -m tools.m8j9_run
```

私有结果目录/root/gpufree-data/hs-drive-private/m8j9-frozen-a-official-eval；逐对象结果、位姿与GT关联只私有保存。公开官方metrics_summary/details是聚合指标及曲线，不含逐对象预测。已有输出不覆盖；后续复现必须使用另一个输出目录并记录路径差异。未对外提交成绩、未合并main。
