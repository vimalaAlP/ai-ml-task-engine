import pandas as pd
df = pd.read_csv('data/Enhaced_ai_ml.csv', parse_dates=['created_at','estimated_deadline','closed_at'])
df.columns = df.columns.str.strip()

# Example fixes:
df['priority'] = df['priority'].fillna('Medium')
num_cols = ['complexity','workload_open_tasks','dependencies_count','story_points','assignee_skill_score',
            'creator_experience_months','comments_count','reopens_count','past_performance_score','actual_time_hours']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())

# Feature: days_to_deadline
df['days_to_deadline'] = (df['estimated_deadline'] - df['created_at']).dt.days.fillna(0).astype(int)

# Save cleaned file
df.to_csv('data/cleaned_dataset.csv', index=False)
