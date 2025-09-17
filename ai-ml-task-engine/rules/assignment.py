def score_assignee(skill_score, workload_open_tasks, past_performance_score, weights=(0.6,0.25,0.15)):
    w_skill, w_workload, w_past = weights
    workload_norm = workload_open_tasks / (workload_open_tasks + 1)
    score = skill_score * w_skill - workload_norm * w_workload + past_performance_score * w_past
    return score

def recommend_assignee(available_assignees):
    best = max(available_assignees, key=lambda a: score_assignee(a['skill_score'], a['workload_open_tasks'], a['past_performance_score']))
    return best['assignee_id'], round(score_assignee(best['skill_score'], best['workload_open_tasks'], best['past_performance_score']), 3)
