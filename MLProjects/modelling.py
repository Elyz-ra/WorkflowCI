# modelling.py (versi perbaikan)

import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

def train_and_log_model():
    """
    Melatih model XGBoost dan mencatatnya ke MLflow menggunakan autolog.
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

    # Cukup aktifkan autolog, tidak perlu log_metric dan log_model manual
    # untuk parameter dasar dan model.
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        registered_model_name="ShippingDelayXGBoostModel" # Kita bisa daftarkan model langsung dari sini
    )

    with mlflow.start_run(): # Praktik terbaik adalah membungkus training dalam run context
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

        print(f"Akurasi model (otomatis dicatat oleh autolog): {acc}")
        print("Model berhasil dicatat dan didaftarkan ke MLflow melalui autolog.")

if __name__ == "__main__":
    train_and_log_model()