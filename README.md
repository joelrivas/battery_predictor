# Battery Life Predictor (Portfolio)

### Objective 
To build a reproducible and scalable pipeline that predicts the remaining battery life of a device per user/interval. The deliverable demonstrates: ingestion, featurization, training, deployment, and monitoring.

### Technical Highlights
- PySpark for large-scale processing (*in progress*)
- Airflow for orchestration (*in progress*)
- LightGBM (or sklearn) for a fast, realistic model
- Basic testing and data checks (*in progress*)

## How to Run Locally (Dev)
Create virtual environment and dependencies: 
    
    python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

Generate sample data: 
    
    python src/ingestion/ingest_events.py --n_users 200 --minutes 1440 10000 --out sample_events

Featurize:

    python src/features/transform.py --in sample_events --out features --battery_threshold 0.10

Train:
    
    python src/train/train_model.py --in data/features.parquet --model-out models/battery_model.pkl

Evaluate:
    
    python src/train/eval.py --model models/battery_model.pkl --test data/features.parquet

Deployment:

    python src/api/app.py --model models/battery_model.pkl (in progress)    

## Run the API

    uvicorn api:app --host 0.0.0.0 --port 8080

### Sample Request

    curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{
        "last_battery": 0.23,
        "mean_battery": 0.66,
        "min_battery": 0.01,
        "max_battery": 1.0,
        "count_events": 412,
        "std_battery": 0.23
    }'


## What to Look for in the Repo

- `src/features/featurize.py`: How to transform event series into features by window (last 24h, slope, variance, average usage).
- `src/train/train_model.py`: Training pipeline with CV (Cross-Validation), metrics (RMSE, MAE), and persistence (saving).
- `airflow/dags/battery_pipeline_dag.py`: DAG for ingest → featurize → train → deploy → monitor. (*in progress*)