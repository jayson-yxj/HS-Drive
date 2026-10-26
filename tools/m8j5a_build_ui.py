"""214-frame local audit UI, shared modality images, no hidden per-object filter."""
import argparse
import json
import os
from pathlib import Path

from datasets.m8j3_rgb_clip import CAMERAS,VIEW
from tools.m8j3_validate_clip import RAW
from tools.m8j5a_audit import OUT,save,report


def technical(o):
    return (o['lineage']['legacy_positive_candidate'] and 'OutsidePETRRange' not in o['filter_reasons']
            and 'NoSupportedViewInPETRCrop' not in o['filter_reasons'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update-review-entry',action='store_true',help='Archive old review page and replace only its navigation entry once')
    args=parser.parse_args()
    os.umask(0o077);web=OUT/'web';web.mkdir(exist_ok=False);images=web/'images';images.mkdir()
    payload=[]
    for file in sorted((OUT/'frames').glob('*.json')):
        f=json.loads(file.read_text());n=f['ordinal'];paths={};objects=[]
        for cam in CAMERAS:
            paths[cam]={}
            for modality in ('rgb','semantic','instance'):
                ext='.jpg' if modality=='rgb' else '.png';name=n+'_'+cam+'_'+modality+ext
                src=RAW/'camera'/(modality+'_'+VIEW[cam])/(n+ext)
                # Read-only source retained; symlink permits local viewing, no copying to Git.
                (images/name).symlink_to(src)
                paths[cam][modality]='images/'+name
        for e in f['objects']:
            objects.append({k:e[k] for k in ('actor_id','source_class','source_state','lineage','filter_reasons','views')})
            objects[-1].update(old_PETR_technical_candidate=technical(e),geometry_reference=e['converted'].get('reference','Unknown'),
                geometry_status=e['converted']['geometry_status'],center_hs_m=e['converted'].get('center_hs_m'),
                dimensions_lwh_m=e['converted'].get('dimensions_lwh_m'))
        payload.append(dict(ordinal=n,objects=objects,images=paths,regions=[r for r in f['render_regions'] if r['pixels']>=100 and r['coverage'].startswith('Unexplained')]))
    page=Path('tools/m8j5a_audit_ui.html').read_text().replace('__DATA__',json.dumps(dict(frames=payload),ensure_ascii=False).replace('<','\\u003c'))
    (web/'index.html').write_text(page)
    # Latest user explicitly requests replacing candidate-only review display.
    oldroot=Path('/root/gpufree-data/hs-drive-private/m8j5-supervision')
    old=oldroot/'review_index.html';backup=oldroot/'review_index_pre_annotation_audit.html'
    if args.update_review_entry:
        if backup.exists():raise FileExistsError('Original review page already archived')
        old.rename(backup)
        # Simple navigation entry, no network redirect or external server.
        from html import escape
        href=escape(os.path.relpath(web/'index.html',oldroot),quote=True)
        old.write_text('<meta charset="utf-8"><h1>Golden GT 签核已暂停</h1><p>已改为全部原始/转换/最终训练 GT 三层审计，不再只画当前候选。</p><p><a href="'+href+'">打开 214 帧三层审计页面</a></p>')
    save(report('m8j5a_ui_status.json'),dict(status='AllObjectsThreeLayerAuditReady',frames=214,
        default_ordinal='00017',default_highlight_actor='394',highlight_hides_others=False,
        layers=['AllRawGT','AllConvertedGT','FinalTrainingGT'],additional_layer='HistoricalPETRTechnicalCandidates_NotFinal',
        final_training_gt=0,golden_signoff='Suspended',server_connections=0,private_page=str(web/'index.html'),
        original_interactive_page_preserved=str(backup) if backup.exists() else None,
        review_entry_updated=args.update_review_entry,images='PrivateLocalSymlinks_NoOriginalMutation'))


if __name__=='__main__':main()
