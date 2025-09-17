import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
from preprocess import preprocessor

df = pd.read_csv('data/cleaned_dataset.csv')
X = df[numeric_features + categorical_features]
y = df['delayed'].astype(int)   # ensure 0/1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1))])
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:,1]
preds = clf.predict(X_test)
print("ROC AUC:", roc_auc_score(y_test, probs))
print("F1:", f1_score(y_test, preds))

joblib.dump({'model': clf, 'model_version': 'risk_v1'}, 'models/risk_predictor.joblib')
