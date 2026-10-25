# M8-D 实现与复现

基线 9af1cb20f1d4fe81b25b80d37001a04a18ae985d，独立分支 feat/m8d-petr-camera-order-ablation。固定官方PETR f7525f93467a33707ef401c587a52d5e7b34de74、900 Query、6层Decoder、10类、FP32。无训练；M8-B/C及默认感知/控制代码只读。

## 输入合同和隔离

A: front, front_left, front_right, rear_left, rear_right, rear。
B: front, front_right, front_left, rear, rear_left, rear_right。
索引[0,2,1,5,3,4]，逆索引[0,2,1,4,5,3]。B名称OfficialConverterOrderCandidate；实际训练PKL顺序未验证。

datasets/m8d_camera_order.py复用原adapter增强，分别执行相同预处理，B仅重排最终camera axis。RGB、名字/路径/SHA、K/T、增强intrinsics、extrinsics、lidar2img、shape/pad、每摄时间联合排列。新增未声明的六元素metadata会拒绝。新增审计metadata两组一致，原模型计算字段不变。192帧在模型运行前逐一检查：图像SHA多重集合、按物理相机对齐的Tensor/K/T/投影矩阵、正负深度解析点投影、逆排列和时间戳。

tools/m8d_gpu.py复用独立PETR环境；unshare --net、环境清理、RGB-only bundle和Python open审计禁止原始GT/ledger/reports。是可信源码监督隔离，不是防恶意native代码沙箱。558张量从审计后的numeric NPZ以allow_pickle=False加载，结束逐位核验。实际模型输入无GT。

12帧保存全部6层900Query分类/回归、位置编码及300候选；192帧关闭Hook、交替A/B先后。两个阶段各首帧每组一次预热；不计预热，steady统计另排除首帧。全部A重放核对历史结果，B不参与规则选择。

## 独立评价

原155GT/38位置/192帧；score>=0.1、XY<=2m且Z<=2m，另4m敏感性。逐GT记录共同成功、丢失、新增、共同失败；空GT位置也报告候选负担。

2D代理沿用原最大基数一对一协议；严格身份复用M8-C实例mask/purity/IoU/多视角规则，无3D误差门槛。所有非IdentitySupported的metric_error为null。严格身份计算XY/XYZ、光学深度/横向、径向/Z、尺寸、yaw/轴向yaw、upright IoU和>5m比例。A/B各自集合、共同集合、固定A六对象分别报告；历史92代理口径单独保留，不能跨分母比较P95当成改善。

Train仅合同开发；Val描述性；Calibration不选顺序/不搜风险阈值；旧Sealed仅历史回归，无新盲测。FP为旧函数字段，含义是Annotation-Unmatched，不是确认物理误检或Free。报告统计按位置而非把192帧视作独立样本。

## 执行与复现

独立输出：/root/gpufree-data/hs-drive-experiments/m8d-camera-order。
环境：/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/env/bin/python。
全部产物排他写入，日志包含源码SHA、返回码、命令和时间。

复现须使用独立checkout和新输出root，保留原protocol及新SHA，旧报告不删除。下列顺序适用于尚未生成M8-D实验产物的独立副本：

~~~bash
python3 - <<'PY'
from tools.m8d_common import *
run('source_audit', [str(RUNTIME/'env/bin/python'), '-m', 'tools.m8d_source_audit'])
run('all_new_tests', [str(RUNTIME/'env/bin/python'), '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_m8d*.py', '-v'])
run('contract192', ['unshare', '--net', str(RUNTIME/'env/bin/python'), '-m', 'tools.m8d_gpu', '--phase', 'contract'])
run('small12', ['unshare', '--net', str(RUNTIME/'env/bin/python'), '-m', 'tools.m8d_gpu', '--phase', 'small'])
run('full192', ['unshare', '--net', str(RUNTIME/'env/bin/python'), '-m', 'tools.m8d_gpu', '--phase', 'full'])
run('evaluate', [str(RUNTIME/'env/bin/python'), '-m', 'tools.m8d_evaluate'])
run('output_audit', [str(RUNTIME/'env/bin/python'), '-m', 'tools.m8d_output_audit'])
run('figures', [str(RUNTIME/'env/bin/python'), '-m', 'tools.m8d_figures'])
PY
~~~

现有交付可运行只读测试：

~~~bash
/root/gpufree-data/hs-drive-experiments/m8a-petr-runtime/env/bin/python -m unittest discover -s tests -p 'test_m8d*.py' -v
~~~

Launcher补齐PETR/PyTorch路径、动态库与PYTHONPATH，详细环境见tools/m8b_common.environment。首次建立新输出需创建protocol指定root；完整复现实验还需对应版本的M8-B/C外置资产。

## 失败与时间口径

首轮small预热后失败：官方extract_feat会原地squeeze输入Tensor，复用时维数不符。修复为每次显式view(1,6,3,512,1408)，不修改模型。原源码、日志和small_failed_v1保留，失败运行没有质量指标。

预处理含PNG读取、官方增强和B复制；H2D、前向、Box decode、D2H、序列化、artifact写入分别计时。E2E截至第一次序列化结束，排除张量压缩/持久化、历史输出读取与额外审计。首对预热从墙钟中剔除，steady统计排除首对。不同口径不能直接解释为性能改善。保留M8-B约338ms E2E和Online NoGo；不认证5Hz/20Hz/共享GPU/CARLA闭环。
