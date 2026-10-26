"""Build self-contained private offline review page. No actual review decisions."""
import base64
import json
import os
from pathlib import Path

from tools.m8j4_sidecar import render


def main():
    os.umask(0o077)
    root=Path(json.loads(Path('configs/m8j5_protocol.json').read_text())['private_root'])
    package=json.loads((root/'golden_candidates.json').read_text())
    directory=root/'interactive_assets';directory.mkdir(exist_ok=False)
    output=[]
    for candidate in package['candidates']:
        folder=directory/candidate['candidate_id'];folder.mkdir()
        render(dict(ordinal=candidate['ordinal'],objects=[candidate['object']]),folder)
        row=dict(candidate)
        for kind in ('rgb','instance'):
            path=folder/(candidate['ordinal']+'_'+kind+'.jpg')
            row[kind+'_image']='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()
        output.append(row)
    template=Path('tools/m8j5_review_ui.html').read_text()
    payload=json.dumps(dict(candidates=output),ensure_ascii=False).replace('<','\\u003c')
    page=template.replace('__REVIEW_DATA__',payload).replace('__DESTINATION__',str(root/'m8j5_review_responses.json'))
    # Preserve previous static page, explicitly requested UI enhancement only.
    original=root/'review_index.html';backup=root/'review_index_static.html'
    if backup.exists():raise FileExistsError('Static original already archived')
    original.rename(backup)
    with original.open('x') as f:f.write(page)
    print(json.dumps(dict(status='PrivateOfflineInteractiveReviewReady',candidates=len(output),bytes=len(page.encode()),
        page=str(original),original_static=str(backup),network_connections=0,actual_signoffs=0,server_write_api=False)))


if __name__=='__main__':main()
