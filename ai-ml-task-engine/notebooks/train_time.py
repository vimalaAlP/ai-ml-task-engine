import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from preprocess import preprocessor

df = pd.read_csv('data/cleaned_dataset.csv')
X = df[numeric_features + categorical_features]
y = df['actual_time_hours']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([('pre', preprocessor), ('reg', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE", mean_absolute_error(y_test, pred))
print("RMSE", np.sqrt(mean_squared_error(y_test, pred)))
print("R2", r2_score(y_test, pred))

joblib.dump({'model': model, 'model_version': 'time_v1'}, 'models/time_estimator.joblib')
