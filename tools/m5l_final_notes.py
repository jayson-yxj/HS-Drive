"""Add explicit evidence-based final limitations without touching frozen science."""
import json
from pathlib import Path
from tools.m5l_common import write,asset

def main():
 support=json.loads(Path('reports/m5l_sealed_support.json').read_text());ref=json.loads(Path('reports/m5l_reference_and_freshness_audit.json').read_text());scene=json.loads(Path('reports/m5l_scene_protection.json').read_text());summary=Path('reports/m5l_summary.md');text=summary.read_text();marker='## 交付复核'
 if marker in text:raise FileExistsError(marker)
 text+='\n'+marker+'\n\n工程接口通过、Vehicle-only训练完成、泛化质量Partial、纯视觉安全NotValidated分别成立，不能互相代替。\n\n'
 text+='原生行人宽度中位约5像素。Train/Val最佳固定P8相位也仅11条，默认相位0条；P4各9条，说明不能仅靠更改相位或插值解决监督稀疏。\n\n'
 text+='Val Adaptive新增Oracle目标端点格929/945，既有目标格跨阈值453/399，交集111/118；背景跨阈值1548/1549。计数按frame-cell累积，有重复/相关性；A/B非互斥。Val同FPR锁点Adaptive阈值0.5，Recall4.08/4.00%，低于Uniform@0.2的5.75/4.44%；因此暂停直接ROI优化升级。\n\n'
 text+='Shadow 30包，28及时输出，1计算前过期、1输出过期；首次推理长尾约930ms，输出完整长尾约1065ms均保留并丢弃。完整P95约133ms；tickP95由70.8增至84.2ms（约19%），未达到预声明停止阈值，但本场景自身平均tick约61ms，不能称20Hz稳定，也不能声称完全没有资源影响。\n\n'
 text+=f"逐位置近场/Bias检查发现{len(scene['failed'])}项未通过，不能用聚合MAE掩盖，见`m5l_scene_protection.json`。这些失败进一步阻止默认升级。\n\n"
 text+='封存集严格支持：\n\n|Stride|类别|20–35m射线|有支持位置|\n|---:|---|---:|---:|\n'
 for r in support['records']:
  if r['band']=='20_35':text+=f"|{r['stride']}|{r['type']}|{r['n']}|{r['positions']}|\n"
 text+='\n目标级方法始终输出contact-unverified条件假设；可信位置数量为0。`contact_supported`仅按GT可靠可见底部端点接近平面诊断，侵蚀后的底部并非轮胎接地点，不能将该诊断当作推理可用的接地证据。坡道采集采用Ego相对配置，地形变化可能导致Actor悬浮/穿地；不宣称精确物理接地真值。\n\n'
 text+='下一阶段建议限定为：RGB可见接地点/接触可信度、可拒绝的目标几何、近场保护及已确认背景抑制。保留Dense分支作背景/表面上下文与车辆研究对照，尚无证据将其直接替换；不自动部署Object Query、SparseDrive或Temporal。\n\n'
 text+='图：[行人采样](m5l_figures/m5l_train_pedestrian_support.png)、[目标XY](m5l_figures/m5l_object_xy.png)、[Adaptive同FPR](m5l_figures/m5l_adaptive_test.png)、[在线Shadow原始时序](m5l_figures/m5l_shadow_timing.png)。\n'
 summary.write_text(text)
 write('reports/m5l_delivery_review.json',dict(kind='postfreeze_readonly_qualification',scene_failures=len(scene['failed']),trusted_object_positions=0,all_outputs_conditioned_on_unverified_contact=True,adaptive_sameFPR_val_gain=False,default_upgrade=False,occupancy_pilot=False,source_reports=[asset('reports/m5l_scene_protection.json'),asset('reports/m5l_sealed_support.json'),asset('reports/m5l_reference_and_freshness_audit.json')]))
if __name__=='__main__':main()
