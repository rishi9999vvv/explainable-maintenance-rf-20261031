# Explainable predictive maintenance

Flask application with pre-trained Random Forest models, temporal vibration features, TreeSHAP, verified endpoint metadata and raw CSV/ZIP replay. Original AI4I demo is mounted at /baseline/.

## Deployment
Python 3.12. Install requirements.txt and start `gunicorn wsgi:application --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180`.

Evaluation is development evidence on one previously inspected bearing. Target is recorded experiment endpoint proximity. Scores are uncalibrated.

Raw third-party XJTU captures, local environments, logs, credentials and private presentation files are excluded. Dataset provenance hashes and extracted feature replay are included.
