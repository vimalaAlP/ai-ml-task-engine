def suggest_actions(tasks):
    actions = []
    for t in tasks:
        if t['priority']=='High' and t['task_status']=='ToDo':
            actions.append({'task_id':t['task_id'],'action':'Assign to top performer','reason':'High priority & ToDo'})
        elif t.get('risk_probability',0) > 0.7:
            actions.append({'task_id':t['task_id'],'action':'Escalate to manager','reason':'High risk'})
        elif t['priority']=='Low' and t['task_status']=='InProgress':
            actions.append({'task_id':t['task_id'],'action':'Send deadline reminder'})
    return actions
