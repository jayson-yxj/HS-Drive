# M8-J6 实际续跑记录与复现

最新报告为 `reports/m8j6_actual_summary.md` 和 `reports/m8j6_actual_acceptance.json`。最初数据阻塞报告保留，未追溯修改。

## 已执行顺序

```bash
.venv/bin/python -m tools.m8j6_ingest \
  --archive '/root/Downloads/v1.0-mini (1).tgz' \
  --output /root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/data \
  --report reports/m8j6_sources/actual_ingest.json

.venv/bin/python -m tools.m8j6_preflight \
  --dataroot /root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/data \
  --private-output /root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/manifest_actual.json \
  --report reports/m8j6_sources/preflight_actual.json

.venv/bin/python -m tools.m8j6_actual_run prepare
# 首次索引问题已保存，修复代理对象后，失败目录保留为 prepared_failed_attempt1。
.venv/bin/python -m tools.m8j6_actual_run prepare --name prepare_retry2

# 查看私有图像并写入真实查看范围后执行：
.venv/bin/python -m tools.m8j6_actual_run baseline
.venv/bin/python -m tools.m8j6_actual_run smoke
# Smoke 返回1；optimizer_steps=1；没有重试。
```

这些命令的输出目录已存在，直接重复将拒绝覆盖。不得删除旧失败或绕过现有停止结果。源码SHA、实际版本、命令、UTC、退出码、日志SHA在各`execution.json`。官方converter与环境未修改。

## 不含私有数据的回归

```bash
.venv/bin/python -m pytest -q tests/test_m8j6_selection.py tests/test_m8j6_ingest.py
.venv/bin/python -m tools.m8j6_check_official --output /tmp/m8j6_alias_repeat
```

第一个20项，第二个16项；第二个明确构造CPU官方模型结构验证共享参数身份，不加载checkpoint，不启动GPU。没有调用主工程Python3.10 conftest到Python3.8环境。

## 私有资产

- `data/`：完整Mini，原始表和图像不修改。
- `manifest_actual.json`：固定八帧、真实相机时间戳及位姿关联。
- `prepared/rgb_input/`：只含规范化RGB、lidar2img及相机时间戳的NPZ。
- `prepared/supervision/`：335条训练GT及484条原始记录逐条过滤追溯，禁止公开上传。
- `prepared/review/index.html`：六摄原始目标、保留状态、BEV及实际训练裁剪框。
- `prepared/visual_review.json`：AI查看者、具体图像、范围及SHA；不是人工签核。
- `smoke/`：首次失败的私有目录；未保存实验权重，不能恢复丢失的第1步状态。

报告中的camera文件路径和sample token用于复现；不包含逐对象3D标签、图片或checkpoint。Hash不代表官方提供过预期摘要。

## 真实停止点和下一次执行条件

共享分支别名修复在 `training/m8j6_frozen_state.py`；实际GPU脚本已接入，但**修复后的真实训练未运行**。最新代码也把反向指标放在step之前保存，失败时另存私有状态，避免再次丢失证据。

下一次须单独确认新的三步预算，并保留当前`smoke/`和`actual_smoke/`。新attempt应从原权重重新开始，仍为索引`[0,4,1]`、FP32、Head-only、AdamW1e-5、weight_decay0，不改变图像/类别/范围；准确声明此前1步与新3步分属不同attempt。

成功前不得运行过拟合。修复后的真实冻结参数/buffer、optimizer状态、checkpoint保存/重载需全部获得实际证据，不能用CPU测试替代。历史M8-J5C AMP/DCNv2失败仍然开放。

## 对初始设计说明的勘误

初始 `M8J6_IMPLEMENTATION.md` 中“浅拷贝对象、只替换.sample而保留索引”的建议在真实执行中不可行：NuScenes.get会访问副本的短sample列表，索引越界。已改为`SelectedMetadataView`，让迭代列表限八帧、所有devkit方法绑定原完整metadata对象。没有因此扩读其他图像或训练帧。
