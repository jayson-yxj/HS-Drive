# M8-J6 实施与续跑

当前实现范围：固定协议、八帧 metadata selector、数据存在性和关联预检、冻结 PETR 官方代码的 CPU 合成合同检查。真实读取、可视化和三步训练因为数据缺失尚未执行，不能用本文件作为其验收证明。

## 本轮可复现命令

在仓库根目录，主工程 Python3.10 环境运行：

```bash
.venv/bin/python -m pytest -q tests/test_m8j6_selection.py
.venv/bin/python -m tools.m8j6_check_official --output /tmp/m8j6_official_contract_repeat
```

第二条复用既有 PETR Python3.8，`unshare --net` 禁网、隐藏 GPU，90秒超时；输出目录必须不存在。不会 pip install 或更改 CUDA/MMCV。

本轮真实缺失数据复现（预期退出码2）：

```bash
.venv/bin/python -m tools.m8j6_preflight \
  --dataroot /root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/data \
  --private-output /root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/recheck_manifest.json \
  --report /tmp/m8j6_preflight_repeat.json
```

输出使用新文件名，拒绝覆盖；私有 manifest 不允许写入仓库。该工具只检查 metadata 与引用文件存在性，不打开图片、点云，不加载模型；返回成功也只代表 `MetadataSelected_NotTrainingGo`。

## 数据到位后的最小执行顺序

1. 用户提供合法 Mini 本地目录，或完成官方登录及条款确认后提供官方获取方式。不需要重新申请三步更新权限。下载如有官方校验值必须核对；没有则仅记录本地 SHA，不称官方认证。仍仅 Mini，不扩展到 trainval。
2. 把真实路径传给上面的 preflight，取得 `scene-0061`、`scene-0655` 各四个最早 sample token、六摄文件、各自 timestamp/calibration/pose 和 annotation 关联。缺失或异常不静默替换。协议在 `configs/m8j6_protocol.json`，已在看到图像和预测前固定。
3. 复用冻结 official converter/dataset/transform，对八帧构建私有 infos 与逐对象过滤追溯。检查 center、full size、Yaw、速度、六摄矩阵及全目标投影；复用已有查看组件生成八帧六摄/BEV 图，记录实际查看者、查看范围、异常。不要求从头人工重标官方数据。
4. 通过后加载冻结 safe tensor archive，batch1完整512×1408六摄 FP32 基线。逐帧逐层保存 P/I/N、有效权重和实际损失归一化。验证 RGB/标定输入与监督分离、无 point/radar/GT 输入、无残余 AMP hook、无半精度模块。任何具体监督异常隔离样本，三步未满足则停止，不临时换帧。
5. 在独立实验模型上仅启用原本可训练的 `pts_bbox_head.*`，不能把 `code_weights` 等原本不可训练参数强制解冻。Backbone/Neck `eval()`、`requires_grad=False`，保存全部参数和 buffer 的前后逐张量对照。Head 正常 train 模式，官方 dropout 保持、固定种子，GridMask 明确禁用。
6. 执行 protocol 的三个 sample 索引 `[0,4,1]`，AdamW lr=1e-5、weight_decay=0、norm clip35。每步先验证分类/回归实际梯度、所有梯度有限、冻结模块未变化，再 step；恰好三步停止。零 decay 避免把衰减充当学习证据。检查 head 参数实际改变及 optimizer moment/step；不要求三步 Loss 单调。
7. 保存私有 checkpoint、optimizer及 RNG 状态；重建同架构加载后，固定 eval 输入按 atol=1e-6/rtol=1e-5 比较，报告误差。原 checkpoint SHA 必须不变。

完整 GPU 输入比 J5C 的64×96合成输入大得多；22GiB 是本轮停止上限，不是显存已实测够用。OOM 或非有限值立即保留失败，不自动缩图、换类、改范围或开启 AMP。既有 AMP/DCNv2 失败不修补，也不妨碍单独验证 FP32。

## 数据与计算状态

[`m8j6_acceptance.json`](../reports/m8j6_acceptance.json) 明确区分：合成接口通过、真实数据未核验、真实基线未执行、三步未执行。
配置的预算不是执行结果；当前没有 checkpoint 保存恢复或 GPU峰值记录。
完成上述真实链路并处理具体异常后，才可提出下一轮小集合过拟合。Mini来自预训练 train 分布，无法证明独立泛化，更不能认证驾驶安全。

禁止提交 Mini图像、逐对象标注、私有清单及实验权重。输出到 `/root/gpufree-data/hs-drive-private/m8j6-nuscenes-mini/`；公开报告只保留汇总、源码、配置、哈希和失败原因。旧 Bench2Drive 资产及历史报告保持只读，不启动 CARLA，不合并 main。
