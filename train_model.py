"""
RainRoute – flood susceptibility model
Trains on flood_dataset_classification.csv and exports flood_model.json for the web app.

Usage:  python train_model.py flood_dataset_classification.csv
Needs:  pandas numpy scikit-learn joblib
"""
import sys, json
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

CSV = sys.argv[1] if len(sys.argv) > 1 else "flood_dataset_classification.csv"
FEATURES = ["Latitude", "Longitude", "Rainfall", "Elevation", "Slope"]   # known for ANY road location
df = pd.read_csv(CSV).drop_duplicates()
print("rows loaded:", len(df))

# ---- 1. Remove label leakage -------------------------------------------------
# 'Disaster Type' == 0 is exactly occured == 1 (flood) -> would give a fake 100% score.
# Deaths / Affected / duration / distance / time only exist AFTER an event -> unusable for prediction.
# Rainfall has two imputed constants, one used only for each class -> also leaks the label.
leak_vals = []
for k in (0, 1):
    vc = df[df.occured == k].Rainfall.round(6).value_counts()
    if vc.iloc[0] > 100:                       # a single value repeated hundreds of times = imputation
        leak_vals.append(vc.index[0])
print("imputed rainfall constants removed:", leak_vals)
df = df[~df.Rainfall.round(6).isin(leak_vals)].reset_index(drop=True)
print("clean rows:", len(df), df.occured.value_counts().to_dict())

X, y = df[FEATURES].values, df.occured.values
# spatial groups: 5-degree cells, so CV tests generalisation to NEW places, not neighbours of training rows
groups = (df.Latitude // 5).astype(int).astype(str) + "_" + (df.Longitude // 5).astype(int).astype(str)
cv = GroupKFold(n_splits=5)

models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "RandomForest": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=0),
}
print("\nSpatial 5-fold CV (unseen regions):")
res = {}
for name, m in models.items():
    p = cross_val_predict(m, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    res[name] = dict(auc=roc_auc_score(y, p), acc=accuracy_score(y, p > .5), f1=f1_score(y, p > .5))
    print(f"  {name:20s} AUC={res[name]['auc']:.3f}  acc={res[name]['acc']:.3f}  F1={res[name]['f1']:.3f}")

# ---- 2. Final model (gradient boosting: small, exportable to JavaScript) ----------
gb = models["GradientBoosting"].fit(X, y)
p = cross_val_predict(GradientBoostingClassifier(**gb.get_params()), X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
print("\nConfusion matrix (spatial CV, rows=true, cols=pred):\n", confusion_matrix(y, p > .5))
imp = dict(zip(FEATURES, gb.feature_importances_.round(3)))
print("Feature importance:", imp)
joblib.dump(gb, "flood_model.joblib")

# ---- 3. Export to JSON for the browser --------------------------------------------
init = float(gb._raw_predict_init(X[:1])[0, 0])
trees = []
for est in gb.estimators_[:, 0]:
    t = est.tree_
    trees.append({"f": t.feature.tolist(), "t": [round(float(v), 4) for v in t.threshold],
                  "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                  "v": [round(float(v), 5) for v in t.value[:, 0, 0]]})
model = {"features": FEATURES, "init": init, "lr": gb.learning_rate, "trees": trees,
         "metrics": {k: {a: round(b, 3) for a, b in v.items()} for k, v in res.items()},
         "importance": {k: float(v) for k, v in imp.items()},
         "trained_on": int(len(df))}
json.dump(model, open("flood_model.json", "w"), separators=(",", ":"))
# reference predictions so the JS implementation can be verified
ref = X[:20]
json.dump({"X": ref.tolist(), "p": gb.predict_proba(ref)[:, 1].tolist()}, open("reference_predictions.json", "w"))
print("\nSaved flood_model.json, flood_model.joblib, reference_predictions.json")
