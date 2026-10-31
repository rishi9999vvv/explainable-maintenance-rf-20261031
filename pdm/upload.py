"""Bounded, in-memory raw CSV parsing. ZIP members are never extracted to disk."""
import io,json,re,zipfile,hashlib
import numpy as np
from .features import extract
from .metrics import warning_states

def parse_upload(files,known_manifest=None):
    entries=[];declared_synthetic=False;endpoint_metadata=None
    def read_endpoint(body):
        nonlocal endpoint_metadata
        if endpoint_metadata is not None:raise ValueError('Upload only one endpoint_metadata.json.')
        if len(body)>8192:raise ValueError('Endpoint metadata must be smaller than 8 KiB.')
        endpoint_metadata=json.loads(body)
        if not isinstance(endpoint_metadata,dict):raise ValueError('Endpoint metadata must be a JSON object.')
    for file in files:
        name=file.filename or ''
        if name.lower().endswith('.zip'):
            if len(files)!=1:raise ValueError('Upload one ZIP or a set of CSV files, not both.')
            try:z=zipfile.ZipFile(file.stream)
            except zipfile.BadZipFile:raise ValueError('Invalid ZIP archive.')
            with z:
                members=[m for m in z.infolist() if not m.is_dir()]
                if len(members)>258 or sum(m.file_size for m in members)>420*1024*1024:raise ValueError('ZIP exceeds the 256-capture / 420 MiB expanded limit.')
                for member in members:
                    if member.flag_bits&1:raise ValueError('Encrypted ZIPs are not supported.')
                    if re.split(r'[/\\]',member.filename)[-1]=='endpoint_metadata.json':
                        if member.file_size>8192:raise ValueError('Endpoint metadata must be smaller than 8 KiB.')
                        read_endpoint(z.read(member));continue
                    if member.filename.endswith('demo_metadata.json'):
                        if member.file_size>8192:raise ValueError('Demo metadata must be smaller than 8 KiB.')
                        metadata=json.loads(z.read(member))
                        if not isinstance(metadata,dict):raise ValueError('Demo metadata must be a JSON object.')
                        declared_synthetic=metadata.get('source_kind')=='synthetic_fixture'
                        continue
                    if not member.filename.lower().endswith('.csv'):raise ValueError('ZIP may contain CSV captures, endpoint_metadata.json and demo_metadata.json only.')
                    if member.file_size>3*1024*1024:raise ValueError('Each capture must be smaller than 3 MiB.')
                    entries.append((member.filename,z.read(member)))
        elif re.split(r'[/\\]',name)[-1]=='endpoint_metadata.json':
            read_endpoint(file.stream.read(8193))
        elif name.lower().endswith('.csv'):
            body=file.stream.read(3*1024*1024+1)
            if len(body)>3*1024*1024:raise ValueError('Each capture must be smaller than 3 MiB.')
            entries.append((name,body))
        else:raise ValueError('Use CSV captures with optional endpoint_metadata.json, or one ZIP.')
    if not 16<=len(entries)<=256:raise ValueError('Upload 16–256 captures; 18 or more are needed for persistent warnings.')
    ordered=[]
    for name,body in entries:
        match=re.search(r'(?:^|[/\\])(\d+)\.csv$',name,re.I)
        if not match:raise ValueError('Name captures 1.csv, 2.csv, … in chronological order.')
        ordered.append((int(match[1]),name,body))
    ordered.sort()
    if [x[0] for x in ordered]!=list(range(1,len(ordered)+1)):raise ValueError('Captures must start at 1.csv with no missing or duplicate indices.')
    rows=[];digests=[]
    for index,name,body in ordered:
        try:
            text=body.decode('utf-8-sig');header=text.splitlines()[0].strip().split(',')
            if header!=['Horizontal_vibration_signals','Vertical_vibration_signals']:raise ValueError('Expected Horizontal_vibration_signals,Vertical_vibration_signals header.')
            capture=np.loadtxt(io.StringIO(text),delimiter=',',skiprows=1)
            features=extract(capture)
        except (UnicodeError,IndexError,ValueError) as exc:raise ValueError(f'{name}: {exc}')
        rows.append({'capture_index':index,'elapsed_seconds':(index-1)*60,'features':features.tolist(),'speed_rpm':2100,'load_kn':12,'partition':'uploaded'})
        digests.append({'filename':name,'sha256':hashlib.sha256(body).hexdigest()})
    endpoint=None
    if endpoint_metadata is not None:
        m=endpoint_metadata;index=m.get('endpoint_index');bearing=m.get('bearing_id')
        if m.get('schema_version')!=1 or m.get('endpoint_kind')!='recorded_experiment_endpoint' or type(index) is not int:
            raise ValueError('Use schema_version 1, an integer endpoint_index and endpoint_kind recorded_experiment_endpoint.')
        if declared_synthetic:raise ValueError('Synthetic fixtures cannot declare a verified experiment endpoint.')
        if not known_manifest or bearing not in known_manifest['counts'] or known_manifest['counts'][bearing]!=index or index!=len(rows):
            raise ValueError('Endpoint metadata must match a complete bearing history from the project dataset.')
        expected={r['capture_index']:r['sha256'] for r in known_manifest['records'] if r['bearing_id']==bearing}
        if any(expected.get(i)!=d['sha256'] for i,d in enumerate(digests,1)):
            raise ValueError('Capture checksums do not match the endpoint metadata bearing. Endpoint was not accepted.')
        endpoint={'endpoint_index':index,'bearing_id':bearing,'endpoint_kind':'recorded_experiment_endpoint',
                  'source':'verified_project_dataset','description':'Recorded experiment endpoint; physical failure onset is not separately verified.'}
    origin='verified_project_dataset' if endpoint else ('user_declared_synthetic' if declared_synthetic else 'user_supplied_unverified')
    return rows,origin,digests,endpoint

def score_uploaded(predictor,rows,bearing,endpoint=None):
    values=np.array([r['features']+[r['speed_rpm'],r['load_kn']] for r in rows])
    for row in rows:row['predictions']={};row['bearing_id']=bearing
    meta=[{'bearing_id':bearing,'capture_index':r['capture_index'],'elapsed_seconds':r['elapsed_seconds']} for r in rows[15:]]
    for name in predictor.models:
        scores,_=predictor.score_values(values,name);threshold=predictor.config['models'][name]['threshold'];active,starts=warning_states(scores,threshold,meta)
        for row in rows[:15]:row['predictions'][name]=None
        for row,score,a,e in zip(rows[15:],scores,active,starts):
            target=None if not endpoint else int(0<endpoint['endpoint_index']-row['capture_index']<=10)
            row['predictions'][name]=None if endpoint and row['capture_index']==endpoint['endpoint_index'] else {'score':float(score),'warning':bool(a),'episode_start':bool(e),'target':target,'partition':'uploaded'}
    return rows
