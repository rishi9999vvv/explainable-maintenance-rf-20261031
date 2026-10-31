"""Download the original public data with provenance; training uses the bundled copy."""
from pathlib import Path
import urllib.request, zipfile, io, hashlib
root=Path(__file__).resolve().parent/'data'
root.mkdir(exist_ok=True)
url='https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip'
with urllib.request.urlopen(url,timeout=120) as response:
    payload=response.read()
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    csv=archive.read('ai4i2020.csv')
(root/'ai4i2020.csv').write_bytes(csv)
print('Downloaded UCI AI4I 2020:',hashlib.sha256(csv).hexdigest())
