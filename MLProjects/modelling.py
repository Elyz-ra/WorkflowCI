import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

def train_and_log_model():
    """
    Melatih model XGBoost dan mencatatnya ke MLflow,
    sekaligus mendaftarkan model.
    """
    print("Memulai proses pelatihan model...")

    # Pastikan direktori dataset ada
    # Dalam konteks GitHub Actions, pastikan folder ini ada di root MLProject
    data_path = "ecommerce_shipping_data_preprocessed"
    if not os.path.exists(data_path):
        print(f"Error: Direktori data '{data_path}' tidak ditemukan.")
        print("Pastikan folder 'ecommerce_shipping_data_preprocessed' berisi file CSV Anda.")
        exit(1)

    # Load dataset yang sudah di-preprocessing
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
        # Mengaktifkan autologging untuk sklearn (termasuk XGBoost)
        mlflow.sklearn.autolog()

        # Inisialisasi dan latih model XGBoost
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

        # Prediksi dan evaluasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Mencatat metrik akurasi
        mlflow.log_metric("accuracy", acc)
        print(f"Akurasi model: {acc}")

        # Secara eksplisit mencatat model dan mendaftarkannya
        # Ini penting agar `mlflow models build-docker` dapat menemukan model yang terdaftar
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
