import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

# Set MLflow Tracking URI dan Experiment
# Ini akan mengarahkan MLflow untuk mencatat ke server lokal
mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("Shipping Delay Prediction")

def train_and_log_model():
    """
    Melatih model XGBoost dan mencatatnya ke MLflow,
    sekaligus mendaftarkan model.
    """
    print("Memulai proses pelatihan model...")

    data_path = "ecommerce_shipping_data_preprocessed"
    if not os.path.exists(data_path):
        print(f"Error: Direktori data '{data_path}' tidak ditemukan.")
        print("Pastikan folder 'ecommerce_shipping_data_preprocessed' berisi file CSV Anda.")
        exit(1)

    try:
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).values.ravel()
        print("Dataset berhasil dimuat.")
    except Exception as e:
        print(f"Error saat memuat dataset: {e}")
        exit(1)

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog()

        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        print("Melatih model XGBoost...")
        model.fit(X_train, y_train)
        print("Model berhasil dilatih.")

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        print(f"Akurasi model: {acc}")

        model_name = "ShippingDelayXGBoostModel"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="xgboost_model",
            registered_model_name=model_name
        )
        print(f"Model '{model_name}' berhasil dicatat dan didaftarkan ke MLflow.")

        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

if __name__ == "__main__":
    train_and_log_model()