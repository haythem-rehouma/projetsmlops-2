import argparse, os, sys, mlflow, mlflow.sklearn
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--model",    required=True, choices=["elasticnet", "ridge", "lasso"])
parser.add_argument("--alpha",    type=float, default=0.5)
parser.add_argument("--l1_ratio", type=float, default=0.5)  # ignoré par Ridge/Lasso
args = parser.parse_args()

# ---------- MLflow ----------
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(f"mlops_redwine_{args.model}")

# ---------- Data ----------
csv_path = "data/red-wine-quality.csv"
if not os.path.exists(csv_path):
    sys.exit(f"Fichier introuvable : {csv_path}")
df = pd.read_csv(csv_path)

X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def calc_metrics(y_true, y_pred):
    return dict(
        rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
        mae =mean_absolute_error(y_true, y_pred),
        r2  =r2_score(y_true, y_pred)
    )

# ---------- Entraînement ----------
with mlflow.start_run():
    if args.model == "elasticnet":
        model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
        mlflow.log_param("l1_ratio", args.l1_ratio)
    elif args.model == "ridge":
        model = Ridge(alpha=args.alpha, random_state=42)
    else:
        model = Lasso(alpha=args.alpha, random_state=42)

    mlflow.log_param("alpha", args.alpha)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    for k, v in calc_metrics(y_test, preds).items():
        mlflow.log_metric(k, float(v))

    mlflow.sklearn.log_model(model, "model")
    print(f"Terminé : {args.model}  alpha={args.alpha}")