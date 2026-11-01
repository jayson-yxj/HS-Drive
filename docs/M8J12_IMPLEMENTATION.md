# M8-J12 复现入口

本功能默认关闭，唯一分支 `feat/sparsedrivev2-b2d-integration`。基线 main `eb69114636ab60cd48fdba3708067b751fd4038c`。源码/模型/采样预算固定于 `configs/m8j12_protocol.json`；路线和窗口固定于 `configs/m8j12_capture.json`。实际采集及模型回放源码 commit 为 `6e6eee7732b56c15f5b2ee44940968712a93fb50`，后续缓存图与报告不改变模型结果。

## 环境与资产

独立根目录 `/root/gpufree-data/hs-drive-experiments/m8j12-sparsedrivev2`：`official-source` 是固定 bench2drive commit；`runtime-source` Python 文件逐一与官方相同，仅添加编译产物；`env` 为 Python3.8推理环境；`capture-env` 为 Python3.10采集环境。

HF 仓库/revision/LFS 大小哈希及唯一权重见 `reports/m8j12_sources/asset_plan.json`。`tools/m8j12_assets.py` 使用固定官方 resolve、有限超时/续传；失败和完成分别记录，不能自动重新下载整个仓库。不要把 proxy 凭据写入 Git。PyTorch 官方 wheel 与 SHA 见 `wheel_download.json`；全部安装命令及确切依赖见 `dependencies_runtime_only.json`、`import_dependencies*.json`、`ipython_import_dependency.json` 和 `installed_packages.txt`。

环境不共用 PETR/StreamPETR site-packages。编译复用只读 CUDA11.8 toolkit，`TORCH_CUDA_ARCH_LIST=8.9`、`MAX_JOBS=4`。初次 serial fallback 因 PATH 未含新环境 ninja 已停止并记录，修正后使用原源码构建成功：

```bash
python3 -m tools.m8j12_phase --timeout 1900 cuda_extensions_path_fixed python3 -m tools.m8j12_build --attempt path_fixed
python3 -m tools.m8j12_phase model_load_operator_call_fixed bash tools/m8j12_env.sh -m tools.m8j12_runtime
```

以上是已执行命令，不可覆盖同名运行日志。`tools/m8j12_env.sh` 使用禁网 namespace、明确 PYTHONPATH/LD_LIBRARY_PATH；未安装不使用的 Jupyter/TensorBoard服务/Triton，不是可直接训练的完整环境。模型以 strict=True 加载唯一完整 checkpoint，参数 dtype FP32，不调用随机初始化/预训练下载。

采集环境通过只读软链接复用现有 CARLA/numpy/Pillow，详见 `capture_environment.json`，没有改动原环境。Pillow 是官方 RoutePlanner debug 构造的必要导入；禁用其显示而不改导航算法。

## 已执行采集与回放

```bash
/root/gpufree-data/hs-drive-experiments/m8j12-sparsedrivev2/capture-env/bin/python -m tools.m8j12_capture --preflight-only
python3 -m tools.m8j12_phase --timeout 850 capture_snapshot_fixed .venv/bin/python -m tools.m8j12_capture_supervisor --attempt snapshot_fixed
python3 -m tools.m8j12_phase --timeout 600 replay_first bash tools/m8j12_env.sh -m tools.m8j12_replay
python3 -m tools.m8j12_phase cache_audit_final_figures bash tools/m8j12_env.sh -m tools.m8j12_replay_audit
```

采集 supervisor 检查端口独占、严格模型加载证据及完整采集导入；创建自己的进程组，先记录实际地图/版本/XODR/argv/cwd，然后专用服务器加载固定 Town03。最多150s启动就绪等待、600s采集墙钟保护；每段模拟时间≤20s，既有 CarlaSession清理Actor/传感器，supervisor清理服务器并核验端口。最终两段各8.3s；前三次失败未取得有效序列，证据不覆盖。任何未来重采集不是本轮自动续跑动作。

输入目录由 `capture_result.json` 绑定；每帧保存真实六摄/IMU frame/timestamp、ego transform、signed forward speed、固定路线当前导航目标、采集动作。无 Actor/地图 GT 标签或点云。每帧官方模型输入严格白名单：`img,timestamp,projection_mat,image_wh,ego_status,gt_ego_fut_cmd,tp_near,img_metas`。导航名称含 gt 不表示目标真值。测量位姿与虚拟传感器固定外参用于时序几何，不创建 LiDAR。

回放前 CARLA 必须停止。FP32、eval/no_grad、TF32关闭；无优化器。每次模型调用前先写私有全局 forward ledger，失败也计数，上限160。只执行了128次。每段重置官方检测/map bank、motion/planning queue、原PID；保存实际历史时间及有效标记。全state_dict及参数版本监测覆盖参数和固定缓冲。

`tools/m8j12_runtime.py` 复用官方 Compose 流水线，仅去掉已由实测图像满足的文件读取步骤。调用官方 JPEG quality20、704×396→704×384 crop、归一化。`tools/m8j12_replay.py` 只读 profiler 获取原解码器局部选中索引/评分，不另写候选选择。相同输入重放以原定1e-5绝对/相对容差核验，实际规划和控制差为0。

`datasets/m8j12_contract.py` 将原生右/前坐标映射HS前/左，保留6点/.5s/3s；速度为窗口平均，yaw为轨迹段派生。`DryRunController`只暴露数值返回，不接受Vehicle/Client，也没有apply_control。真实采集控制器可执行，但它不能读取新模型输出。两条代码路径和进程分离。

私有根目录 `/root/gpufree-data/hs-drive-private/m8j12-integration-20261101` 保存原始观测、输出、权重之外的日志/图/候选；权重在独立runtime根目录。`final_integrity.json`记录私有清单哈希，不公开逐对象内容。

## 测试与可复用范围

```bash
.venv-gpu/bin/python -m pytest tests/m8j12 --confcutdir=tests/m8j12 -q
bash tools/m8j12_env.sh -m pytest tests/m8j12 --confcutdir=tests/m8j12 -q
```

各16passed，无skip。官方source缺失时两个源码对照测试明确skip，其余核心测试不需私有图像；真实GPU算子/严格加载和真实缓存审计另列，不冒充纯合成测试。核心边界：左右轴向、时间不扩展、真实同步、输入白名单、原PID数值/复位、禁止执行输出、官方记忆重置、固定预算/采样，以及虚拟环境和spawn快照回归。

现存同名目录使重跑拒绝覆盖，避免覆盖失败证据。若未来明确批准复现，应在同一功能代码上使用新 run_id/私有根目录、冻结其配置和预算，不另外创建阶段分支。已经完成的128次不能通过新目录绕过预算再次执行。本轮剩余32次未使用，已结案，不自动消费。

结论与原始失败在 `reports/m8j12_summary.md`、`m8j12_acceptance.json`。会话交接的只读完整性复核见 `reports/m8j12_sources/handoff_verification.json`；没有重跑实验或覆盖既有执行记录。合并只整合代码和证据，PETR-A参考、默认模型、运行模式和安全状态不变；main合并与唯一分支清理等待明确确认。交接时远程仅有基线 main，本功能分支仅在本地；当前拟执行清理仅涉及合入后的本地功能分支。
