from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app import create_app
from run_baseline import module as baseline_module
from pathlib import Path
baseline_module.app.template_folder = str(Path(__file__).parent / 'ai4i_baseline' / 'templates')
application = DispatcherMiddleware(create_app(), {"/baseline": baseline_module.app})
