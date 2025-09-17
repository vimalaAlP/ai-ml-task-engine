from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = ['complexity','workload_open_tasks','dependencies_count','story_points',
                    'assignee_skill_score','creator_experience_months','comments_count',
                    'reopens_count','past_performance_score','days_to_deadline']
categorical_features = ['task_type','priority']

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='NA')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([('num', num_pipeline, numeric_features), ('cat', cat_pipeline, categorical_features)])
