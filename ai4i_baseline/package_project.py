"""Package the submission without machine-specific environments or private builds."""
from pathlib import Path
import zipfile,hashlib,json
ROOT=Path(__file__).resolve().parent
out=ROOT.parent/'AUCML_Submission.zip'
excluded={'.venv','.packages','.build','__pycache__','.git'}
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(part in excluded for part in p.relative_to(ROOT).parts) and not p.name.endswith('.log')]
manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='MANIFEST_SHA256.json'}
(ROOT/'MANIFEST_SHA256.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in files:
        if p.name!='MANIFEST_SHA256.json':archive.write(p,Path('AUCML')/p.relative_to(ROOT))
    archive.write(ROOT/'MANIFEST_SHA256.json','AUCML/MANIFEST_SHA256.json')
print(out,len(files),'files',out.stat().st_size,'bytes')
