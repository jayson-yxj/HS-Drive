# HS-Drive 开发与分支约定

本约定采用用户于 2026-10-29 明确调整的分支策略，替代“每个实验阶段都新建分支、长期不合并 main”的默认做法。历史任务书中的阶段边界和研究结论保持原样；其中的旧分支管理方式不再自动沿用到新任务。

`main` 是可复现研究代码主线，不表示所有模型、实验或驾驶安全均通过。合并工具代码与采用某份权重、启用某种控制模式是不同决定。PETR-A 继续作为当前感知比较参考；StreamPETR 仍是研究候选。默认视觉模型、Planner、Safety、Controller 和 Physics 不因分支整合而改变。

## 日常开发

- 一个可独立交付的功能使用一个短期分支，命名描述功能，而非机械地按阶段编号创建。
- 同一功能的修复、补充验证、实验变体优先沿用现有分支，以 commit、唯一 run_id 和冻结配置区分。
- 功能交付后，以代码兼容性、相关回归、默认行为和复现说明判断合并；模型未超过 A、实验为 Partial/No-Go、完整驾驶未通过，不构成工具代码不能合并的理由。
- 合并前保持明确的配置入口与依赖隔离。CARLA/HS、PETR、StreamPETR 的不同 Python/MMCV/MMDetection3D 环境不混装，不要求它们共享一个全量 pytest 进程。
- 公开历史不 rebase、squash 或强推。线性关系可快进；分叉功能保留原提交，通过普通 merge 整合。
- 仅对少量可复现里程碑使用注释 tag，tag 文案记录实际能力范围和未通过项。实验编号、配置和 run_id 不必各有一个 tag。
- 删除分支前检查准确 tip 已被 main 包含、远程没有新提交、没有仍在使用的 worktree/PR；用明确名单，不批量强删。原 SHA、失败记录及实验结论继续可追溯。

## 研究复现与资产

每次实际实验至少记录源码 commit、必要的源码哈希、配置哈希、run_id、数据/权重标识与哈希、环境版本、命令、退出码及结果限制。新运行使用独立输出目录，不覆盖旧失败。旧脚本对特定 SHA 的检查仍有效；复现旧实验时按 SHA 使用 detached worktree，并遵守原实验运行范围，不能把 checkout 当作新的训练或 CARLA 授权。

原图、数据载荷、逐对象私有标签/预测和权重继续保存在私有目录。Git 只新增代码、配置、聚合指标和经检查的复现证据。已有公开历史中的大文件不在这次整理中删改历史或迁移到 LFS；删除分支也不会自动减少仍由 main 引用的对象体积。

## 一次性整合

具体原始 SHA、65 个功能分支名单、4 个 tag 提议、必要回归及执行顺序见 [整合清单](reports/branch_consolidation_plan.md)。本次用户要求：**实际更新 main 和删除远程分支，须按具体清单确认后统一执行**。当前审查不代替这项确认，也不授权新的研究运行。

## 测试入口

本次审查复用了原 main 的全部 23 个测试文件及研究工具的相关合同测试。准确命令和各环境结果见 [命令](reports/branch_consolidation_logs/commands.json)、[执行记录](reports/branch_consolidation_logs/execution.json)、[补充记录](reports/branch_consolidation_logs/retry_execution.json)。

- `.venv-gpu`：HS 数据、几何、监督隔离与 CPU 合成检查；规划几何需按历史说明设置 `PYTHONPATH=runs/m7a_geometry_deps:.`。
- `.venv`：CARLA 0.9.15 SDK 数学/模拟对象检查，不启动服务器。
- `tools/m8j11_baseline_env.sh`：隔离 PETR 环境，`--noconftest` 避免加载 Python 3.10 的根 conftest。
- `tools/m8j11_env.sh`：隔离 StreamPETR 环境，使用 `--confcutdir=tests/m8j11`。

历史 `scripts/bootstrap_sim.sh` 中的无选择全量 pytest 早于这些隔离环境，不能作为所有研究工具共享一个环境的安装承诺。本次不安装/升级依赖，也不修改默认启动脚本；新功能只执行其依赖环境内的相关回归。测试记录必须如实保留失败及 skip，不能用研究质量状态代替接口测试结果。
