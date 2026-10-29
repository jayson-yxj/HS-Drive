# 一次性分支整合清单（待用户确认）

这是提交关系与代码整合审核，不是新增研发阶段。来源为实际远程 refs、Git 祖先关系、本机 worktree、GitHub 开放 PR 与限定离线回归。完整 SHA 见 [inventory.json](branch_consolidation/inventory.json)，删除名单见 [delete_after_merge.tsv](branch_consolidation/delete_after_merge.tsv)。

核对时间：2026-10-29T16:49:34.375734+00:00。main=`0f8eab324302d35d65095d097d63a3962b055481`，研究代码顶端=`6bdfaaa770380886ef06b27db100b9eacdc5adb1`。

## 实际关系

66个远程分支（main + 65个功能/阶段分支），66个本地分支与远程一一相同。208个后续提交是单一祖先链，没有 merge commit，没有独立分叉；64个旧研究分支均被当前研究顶端包含。当前分支相对所有其他分支仅有最后8个独有提交；相对main有208个提交。这里“独有”明确区分比较对象。

全部479个main原有文件在研究顶端保持相同Git blob/mode；后续新增17,138个文件，没有修改或删除main原有文件。包含历史工具、配置、报告及测试，不需逐个合并65个分支，也不需cherry-pick或squash。

唯一worktree使用当前分支；API检查没有开放PR；未发现正在使用历史分支的本机研究进程。配置中的旧分支名主要是历史来源文字，未发现执行程序依赖这些分支名仍存在。部分历史脚本要求准确旧SHA或独占输出目录，今后应按原SHA detached checkout/worktree复现，不删这些约束。其他机器/未可见自动化的使用状态未知，执行前再检查。

## 统一执行建议

1. 当前分支承载本次政策、文档和审查记录，不再新建阶段分支。完成后的审查提交是唯一候选目标；确认时绑定该完整SHA。
2. 经用户确认后，将main从上述旧SHA一次 `--ff-only` 前进到审查提交。所有历史SHA保持不变，不变基、不压缩、不强推；若远程main或任何待删分支变化，停止并更新差异，不能静默纳入新提交。
3. 创建下面4个注释tag。其余阶段由本清单的原始SHA定位，不逐阶段造tag。
4. 确认远程main已含每个待删tip后，逐项按下面准确名单删除65个远程功能分支；当前工作分支最后删除，并先切到已更新main。检查PR/worktree，任一分支有新工作即保留。按名单记录每次结果，不用通配符删分支、不使用 `-D`，不运行GC/历史改写。对应本地分支可在main可达性检查后用 `git branch -d` 清理。
5. 清理后保留main，今后仅保留正在交付的短期功能分支。同一功能修复和实验变体以commit/run_id/config区别。不会因模型未超过A或驾驶未通过而阻止合入工具；模型晋升与默认模式变化必须另有明确决定。

当前尚未移动main、创建tag、删除本地/远程分支或强推。用户本次要求具体清单确认后统一执行，因此这里只准备完整可审查结果。

## 少量tag提议

| Tag | 固定提交 | 含义 |
|---|---|---|
| `research/m5-closeout` | `7ec866ff4ec5fe621c506056f41750651adcc4ab` | M5 ResearchCompleted / CandidateNoGo / SafetyNotValidated |
| `research/oracle-offline-baseline` | `b3dab191485dfe6da883d0d361064f02522b1a9c` | M7 offline planner research; OnlinePlannerNoGo / SafetyNotValidated |
| `research/petr-fp32-smoke` | `f4e44f9527f9a2869fc1711ff5051d558417beeb` | Exactly three FP32 updates and restore; no generalization certification |
| `research/frozen-perception-comparison` | `6bdfaaa770380886ef06b27db100b9eacdc5adb1` | Frozen A and StreamPETR comparison; A remains reference; no default replacement |

## 精确分支清理清单

下表“main之后”是相对main的累计提交数；“当前之外”是未被研究顶端包含的提交数。所有删除操作均须先完成确认的main整合。表中末行当前分支的远程tip若承载并推送本次审查提交，应以批准的审查提交替换，其他64项不得漂移。

| 分支 | 审计时tip | main之后 | 当前之外 | 使用/动作 |
|---|---|---:|---:|---|
| `main` | `0f8eab324302` | 0 | 0 | 保留main |
| `feat/m45-geometry-foundation` | `129a4131567a` | 1 | 0 | 历史；整合后清理 |
| `feat/m46-occupancy-motion-quality` | `e9e2e02097a7` | 2 | 0 | 历史；整合后清理 |
| `feat/m47-label-assurance-dynamics` | `d8393bc34147` | 3 | 0 | 历史；整合后清理 |
| `feat/m48-spatial-evidence-control` | `ffb466df3471` | 4 | 0 | 历史；整合后清理 |
| `feat/m5a-visual-evidence-generalization` | `a252a04723a2` | 5 | 0 | 历史；整合后清理 |
| `feat/m5b-counterfactual-occupancy` | `dbb70ed0dbd5` | 6 | 0 | 历史；整合后清理 |
| `feat/m5c-obstacle-localization` | `2aa91f6a0458` | 7 | 0 | 历史；整合后清理 |
| `feat/m5d-visible-existence` | `33005efa8ad0` | 9 | 0 | 历史；整合后清理 |
| `feat/m5e-far-visibility-depth` | `bf989b466640` | 11 | 0 | 历史；整合后清理 |
| `feat/m5f-image-grounding` | `5e44a77701f5` | 13 | 0 | 历史；整合后清理 |
| `feat/m5g-visual-representation-depth` | `bf0c38d49e70` | 15 | 0 | 历史；整合后清理 |
| `feat/m5h-distill-near-depth` | `4e5020360237` | 18 | 0 | 历史；整合后清理 |
| `feat/m5i-visual-transfer-depth-qualification` | `3c9c8ad9a701` | 24 | 0 | 历史；整合后清理 |
| `feat/m5j-mobile-bev-evidence` | `db03730a64c0` | 27 | 0 | 历史；整合后清理 |
| `feat/m5k-target-depth-geometry` | `0175f219e555` | 29 | 0 | 历史；整合后清理 |
| `feat/m5l-sparse-geometry-feasibility` | `f2f4e1a79355` | 32 | 0 | 历史；整合后清理 |
| `feat/m5m-selective-object-geometry` | `29ad67af6a21` | 33 | 0 | 历史；整合后清理 |
| `feat/m5n-direct-actor-localization` | `2afe740be0cf` | 34 | 0 | 历史；整合后清理 |
| `feat/m5o-perception-benchmark-reset` | `349941e61abf` | 50 | 0 | 历史；整合后清理 |
| `feat/m6a-system-baseline` | `7ec866ff4ec5` | 52 | 0 | 历史；整合后清理 |
| `feat/m6b-oracle-maneuvers` | `f047ade6d780` | 62 | 0 | 历史；整合后清理 |
| `feat/m6c-safe-stopping-dynamics` | `d808c21c1a4a` | 88 | 0 | 历史；整合后清理 |
| `feat/m6d-signed-stop-control` | `ba5ba6e759b8` | 94 | 0 | 历史；整合后清理 |
| `feat/m6e-actuation-physics-feasibility` | `b0ee24923641` | 99 | 0 | 历史；整合后清理 |
| `feat/m6f-vehicle-dynamics-observability` | `51b410942428` | 109 | 0 | 历史；整合后清理 |
| `feat/m6g-wheel-dynamics-causality` | `eb2c8f409632` | 114 | 0 | 历史；整合后清理 |
| `feat/m7a-scenario-shadow-benchmark` | `3d149ac872d2` | 119 | 0 | 历史；整合后清理 |
| `feat/m7b-dynamic-planning-robustness` | `684b6e64bfd7` | 122 | 0 | 历史；整合后清理 |
| `feat/m7c-planning-efficiency-coverage` | `d9899fac94f3` | 127 | 0 | 历史；整合后清理 |
| `feat/m7d-oracle-temporal-replay` | `6ed646ccffba` | 128 | 0 | 历史；整合后清理 |
| `feat/m7e-state-anchored-planning` | `b3dab191485d` | 129 | 0 | 历史；整合后清理 |
| `feat/m8-0-architecture-review` | `478070b51f03` | 130 | 0 | 历史；整合后清理 |
| `feat/m8a-petr-p0-feasibility` | `e2668bff247b` | 133 | 0 | 历史；整合后清理 |
| `feat/m8a-proxy-asset-retry` | `f4c49a5dbf20` | 139 | 0 | 历史；整合后清理 |
| `feat/m8a-petr-forward-compat` | `a3a892219e29` | 152 | 0 | 历史；整合后清理 |
| `feat/m8b-petr-carla-zero-shot` | `af4266be5859` | 153 | 0 | 历史；整合后清理 |
| `feat/m8c-petr-reference-distribution-audit` | `9af1cb20f1d4` | 154 | 0 | 历史；整合后清理 |
| `feat/m8d-petr-camera-order-ablation` | `037ddba29d6e` | 155 | 0 | 历史；整合后清理 |
| `feat/m8e-perception-data-groundtruth` | `12bd124b443c` | 156 | 0 | 历史；整合后清理 |
| `feat/m8f-grounded-scene-quality` | `8cde984ee606` | 157 | 0 | 历史；整合后清理 |
| `feat/m8g-placement-feasibility-diagnostic` | `410d60b3449e` | 158 | 0 | 历史；整合后清理 |
| `feat/m8h0-native-scenario-source-review` | `5e4701c0cdb8` | 159 | 0 | 历史；整合后清理 |
| `feat/m8h1-native-template-actor-smoke` | `69ce3158ab69` | 160 | 0 | 历史；整合后清理 |
| `feat/m8i0-existing-scenario-data-review` | `502f0701025f` | 162 | 0 | 历史；整合后清理 |
| `feat/m8i1-traffic-manager-development-pilot` | `916ba7507225` | 164 | 0 | 历史；整合后清理 |
| `feat/m8i1a-startup-identity-diagnostic` | `5c14c347de2b` | 166 | 0 | 历史；整合后清理 |
| `feat/m8i1b-offline-startup-forensics` | `fe8db837d18f` | 167 | 0 | 历史；整合后清理 |
| `feat/m8j0-bench2drive-license-access-review` | `307f4fb34b50` | 169 | 0 | 历史；整合后清理 |
| `feat/m8j1-bench2drive-permission-access-closure` | `5d6252a6f5a5` | 170 | 0 | 历史；整合后清理 |
| `feat/m8j2-bench2drive-proxy-access-recheck` | `c4679352669e` | 171 | 0 | 历史；整合后清理 |
| `feat/m8j3-bench2drive-single-clip-validation` | `86984a6843a0` | 172 | 0 | 历史；整合后清理 |
| `feat/m8j4-single-frame-gt-forward` | `f8e9f9ca5f57` | 173 | 0 | 历史；整合后清理 |
| `feat/m8j5-ignore-aware-petr-supervision` | `4042794aa6b1` | 174 | 0 | 历史；整合后清理 |
| `feat/m8j5-raw-annotation-audit` | `6ee9bf7af7b2` | 175 | 0 | 历史；整合后清理 |
| `feat/m8j5b-coverage-aware-supervision` | `92ee99ab68d3` | 176 | 0 | 历史；整合后清理 |
| `feat/m8j5b-loss-validation` | `41430113f389` | 177 | 0 | 历史；整合后清理 |
| `feat/m8j5c-numerical-supervision-closure` | `ffa9b0e6b733` | 178 | 0 | 历史；整合后清理 |
| `feat/m8j6-nuscenes-mini-fp32-smoke` | `2d75f38cd648` | 180 | 0 | 历史；整合后清理 |
| `feat/m8j6-fp32-resume-smoke-retry` | `f4e44f9527f9` | 182 | 0 | 历史；整合后清理 |
| `feat/m8j7-eight-frame-fp32-overfit` | `f50aead109c4` | 185 | 0 | 历史；整合后清理 |
| `feat/m8j8-residual-independent-eval` | `93807325a761` | 187 | 0 | 历史；整合后清理 |
| `feat/m8j8a-cache-transfer-closeout` | `136023d7cda9` | 190 | 0 | 历史；整合后清理 |
| `feat/m8j9-frozen-a-official-eval` | `eb6124603140` | 194 | 0 | 历史；整合后清理 |
| `feat/m8j10-multiscene-400step-control` | `68ee3e80397a` | 200 | 0 | 历史；整合后清理 |
| `feat/m8j11-streampetr-r50-frozen-candidate` | `6bdfaaa77038` | 208 | 0 | 当前worktree；整合后最后清理 |

## 必要回归与整合判断

| 检查 | 本次实际结果 | 范围与限制 |
|---|---|---|
| 原main全部23个测试文件，现有HS CPU环境 | 111 passed / 6 skipped | 5项SDK依赖、1项CUDA；非训练重跑 |
| 原CARLA SDK数学及模拟对象补查 | 5 passed | 补齐上述5项SDK skip，没有启动服务器 |
| 研究合同、Oracle/视觉隔离、规划几何、数据/指标 | 182 passed | 20个指定文件；非完整历史实验重演 |
| PETR别名/冻结/恢复、官方导出、分组/对照 | 43 passed | 隔离PETR CPU环境，合成单元测试 |
| StreamPETR坐标/时序合同与官方框定义 | 10 passed / 1 skipped | CUDA Attention专用检查未重复执行 |

初次规划测试收集失败原因是没有设置旧文档指定的Shapely隔离目录；保留原日志，并按既有环境设置补跑通过。未升级依赖、改变测试或放宽容差。上述组有重叠，不将计数相加宣称独立测试总数。两项CUDA专用测试这次未执行，不把历史GPU结果冒充本次重跑。日志、命令与SHA见 [回归结果](branch_consolidation/regression_results.json)。

提交空白检查发现PETR原始warning日志的5处行尾空格；保留原始字节及日志SHA，仅对该原始日志以外的提交文件执行空白检查。这不是测试失败，也不修改历史或新运行原始日志来掩盖检查结果。

**整合建议：代码可按一次快进纳入main，待用户批准具体操作。** 理由是完整祖先包含、默认入口兼容与必要回归；不是模型质量达到某个阈值，也不是驾驶系统通过验收。可复现性仍依赖分离的HS/PETR/StreamPETR环境及私有资产，不能宣传单一venv就能运行所有实验。相关默认文件Git blob对照见 [默认兼容性](branch_consolidation/default_compatibility.json)。

本次仅新增组织政策、清单和离线测试记录，README更新项目导航。旧报告、失败、默认代码、配置和模型不改。历史文档中的“不合并main”保留为当时事实；新开发按用户本次明确政策执行。所有研究Partial/No-Go保持原结论。

## 资产与清理范围

当前研究tip有17,617个跟踪文件，展开约4.78GB；历史已包含690个PNG、43个JPG、212个NPZ和3个BIN等报告资产，没有跟踪的pt/pth/ckpt/safetensors权重。不能笼统宣称历史Git只有源码，也不在这次修改公开历史。删除旧分支只减少引用，不会减少仍被main包含的大文件对象。旧来源/许可记录继续保留；本次不新增任何私有图像、逐对象标签/预测或权重，不删除服务器实验目录、数据缓存、环境或私有检查点。

对已公开历史文件不进行LFS迁移、filter-repo、GC或重打包。后续新实验的大体积原始日志和载荷放私有目录，Git保存聚合结论、完整来源SHA和复现入口。这是存储约定，不是附加研究阶段或新增全量审核要求。

## 执行前后检查（同一次整理内）

批准时以审查提交的完整SHA为main目标，旧main必须仍为清单内SHA。65个远程功能分支严格对应TSV，当前分支允许的唯一tip变化是本次批准的审查提交；其他漂移立即停止，不使用强制推送。main先用快进整合并正常推送，核实远程main包含全部65个准确tip后才删除分支。tag名称若已存在且指向不同对象，停止而非覆盖。

删除时逐项保存远程旧SHA/结果；失败项保留，不用通配符强删或自动改变清单。切换main后只用 `git branch -d` 删除相应已合入的本地分支。最后核对main目标SHA、tag目标、远程和本地剩余分支及干净工作区。原tip清单可在需要时恢复准确历史分支；tag不替代原SHA清单。

执行不包含新训练、CARLA启动、默认权重晋升、远程消息/Issue或额外功能开发。用户确认后即可按此清单完成一次整理，不另设连续阶段。

## 提交关系复核命令（只读）

```bash
git ls-remote --heads --tags origin
git worktree list --porcelain
git rev-list --count 0f8eab324302d35d65095d097d63a3962b055481..6bdfaaa770380886ef06b27db100b9eacdc5adb1
git rev-list --count --merges 0f8eab324302d35d65095d097d63a3962b055481..6bdfaaa770380886ef06b27db100b9eacdc5adb1
git diff --name-status --no-renames --diff-filter=MD 0f8eab324302d35d65095d097d63a3962b055481 6bdfaaa770380886ef06b27db100b9eacdc5adb1
# 针对清单中任一准确tip：
git merge-base --is-ancestor 129a4131567ac02edf88b19b9199cad979ce3d72 6bdfaaa770380886ef06b27db100b9eacdc5adb1
```

审计快照的208个提交不含随后封存这份政策和清单的组织提交。最终批准目标应是包含该组织提交的完整SHA（在交付消息给出）；当前远程研究分支仍可停留在原6bdfaaa，不必先推送或新增分支才能审查。研究tip的4个tag目标不因此漂移。
