from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
from datetime import datetime, timezone

app = FastAPI(title="AI-ML Task Engine")

# Load models
time_bundle = joblib.load('models/time_estimator.joblib')
time_model = time_bundle['model']
time_version = time_bundle.get('model_version','time_v1')

risk_bundle = joblib.load('models/risk_predictor.joblib')
risk_model = risk_bundle['model']
risk_version = risk_bundle.get('model_version','risk_v1')

# Schemas for requests
class PredictTimeReq(BaseModel):
    task_id: str
    complexity: float
    workload_open_tasks: float
    dependencies_count: float
    story_points: float
    assignee_skill_score: float
    creator_experience_months: float
    comments_count: float

class PredictRiskReq(BaseModel):
    task_id: str
    complexity: float
    workload_open_tasks: float
    dependencies_count: float
    reopens_count: int
    comments_count: int
    past_performance_score: float
    priority: str
    estimated_deadline: str

# ... Schemas for assign/re-shuffle/automation omitted for brevity (mirror your contract)

def iso_now():
    return datetime.now(timezone.utc).isoformat()

@app.post("/predict/time")
def predict_time(req: PredictTimeReq):
    row = [{
        'complexity': req.complexity,
        'workload_open_tasks': req.workload_open_tasks,
        'dependencies_count': req.dependencies_count,
        'story_points': req.story_points,
        'assignee_skill_score': req.assignee_skill_score,
        'creator_experience_months': req.creator_experience_months,
        'comments_count': req.comments_count,
        'reopens_count': 0,
        'past_performance_score': 0.5,
        'days_to_deadline': 7,
        'task_type': 'Feature',
        'priority': 'Medium'
    }]
    import pandas as pd
    X = pd.DataFrame(row)
    pred = float(time_model.predict(X)[0])
    # naive confidence:
    confidence = 0.8
    return {"task_id": req.task_id, "predicted_hours": round(pred,2), "confidence": confidence, "model_version": time_version, "timestamp": iso_now()}

@app.post("/predict/risk")
def predict_risk(req: PredictRiskReq):
    row = [{  # map fields similar to training
        'complexity': req.complexity,
        'workload_open_tasks': req.workload_open_tasks,
        'dependencies_count': req.dependencies_count,
        'reopens_count': req.reopens_count,
        'comments_count': req.comments_count,
        'past_performance_score': req.past_performance_score,
        'priority': req.priority,
        'estimated_deadline': req.estimated_deadline,
        'task_type': 'Feature',
        'days_to_deadline': 3
    }]
    import pandas as pd
    X = pd.DataFrame(row)
    prob = float(risk_model.predict_proba(X)[0][1])
    level = "High" if prob>=0.75 else ("Medium" if prob>=0.4 else "Low")
    action = "Escalate" if level=="High" else ("Flag" if level=="Medium" else "Monitor")
    return {"task_id": req.task_id, "risk_probability": round(prob,2), "risk_level": level, "recommended_action": action, "model_version": risk_version, "timestamp": iso_now()}

# Implement /assign/recommend, /priority/reshuffle, /automation/suggestions using rules/*.py functions

