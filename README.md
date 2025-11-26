# Battery Life Predictor (Portfolio)

### Objective 
To build a reproducible and scalable pipeline that predicts the remaining battery life of a device per user/interval. The deliverable demonstrates: ingestion, featurization, training, deployment, and monitoring.

### Technical Highlights
- PySpark for large-scale processing (*in progress*)
- Airflow for orchestration (*in progress*)
- LightGBM (or sklearn) for a fast, realistic model
- Basic testing and data checks (*in progress*)

### How to Run Locally (Dev)
Create virtual environment and dependencies: 
    
    python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

Generate sample data: 
    
    python src/ingestion/ingest_events.py --out data/sample_events.parquet --n 10000

Featurize:

    python src/features/featurize.py --in data/sample_events.parquet --out data/features.parquet

Train:
    
    python src/train/train_model.py --in data/features.parquet --model-out models/battery_model.pkl

Evaluate:
    
    python src/train/eval.py --model models/battery_model.pkl --test data/features.parquet

Deployment:

    python src/api/app.py --model models/battery_model.pkl (in progress)    

## What to Look for in the Repo

- `src/features/featurize.py`: How to transform event series into features by window (last 24h, slope, variance, average usage).
- `src/train/train_model.py`: Training pipeline with CV (Cross-Validation), metrics (RMSE, MAE), and persistence (saving).
- `airflow/dags/battery_pipeline_dag.py`: DAG for ingest → featurize → train → deploy → monitor. (*in progress*)