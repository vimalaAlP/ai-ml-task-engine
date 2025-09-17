import pandas as pd
from datetime import datetime

def urgency_score(estimated_deadline, risk_probability, dependencies_count):
    days_to_deadline = (pd.to_datetime(estimated_deadline) - datetime.utcnow()).days
    days_score = 1/(1 + max(days_to_deadline,0) + 0.1)
    dep_score = dependencies_count / (dependencies_count + 1)
    score = 0.6*days_score + 0.3*risk_probability + 0.1*dep_score
    return score

def reshuffle(tasks):
    for t in tasks:
        t['urgency'] = urgency_score(t['estimated_deadline'], t.get('risk_probability',0), t.get('dependencies_count',0))
    tasks_sorted = sorted(tasks, key=lambda x: x['urgency'], reverse=True)
    out = []
    for t in tasks_sorted:
        new_pr = 'High' if t['urgency'] > 0.5 else ('Medium' if t['urgency'] > 0.2 else 'Low')
        out.append({'task_id': t['task_id'], 'new_priority': new_pr, 'reason': ('Near deadline' if new_pr=='High' else '')})
    return out
