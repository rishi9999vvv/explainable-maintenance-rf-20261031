"""Launch the unchanged submitted AI4I application on its own local port."""
from pathlib import Path
import importlib.util, sys
baseline=Path(__file__).parent/'ai4i_baseline'
sys.path.insert(0,str(baseline))
spec=importlib.util.spec_from_file_location('ai4i_demo',baseline/'app.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if __name__=='__main__': module.app.run(host='127.0.0.1',port=5056,debug=False)
