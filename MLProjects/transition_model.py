import mlflow
from mlflow.tracking import MlflowClient

# Pastikan MLflow Tracking URI sesuai dengan tempat MLflow UI Anda berjalan
# Ini harus sama dengan yang ada di modelling.py dan yang Anda gunakan untuk menjalankan mlflow ui
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Inisialisasi MLflow Client
client = MlflowClient()

model_name = "ShippingDelayXGBoostModel"
model_version = 1

# Stage tujuan
target_stage = "Production"

print(f"Mencoba mentransisikan '{model_name}' Versi {model_version} ke stage '{target_stage}'...")

try:
    # Transisikan stage model
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=target_stage
    )
    print(f"Model '{model_name}' Versi {model_version} berhasil ditransisikan ke stage '{target_stage}'.")
    print("Silakan cek MLflow UI untuk verifikasi.")

except Exception as e:
    print(f"Gagal mentransisikan stage model: {e}")
    print("Pastikan MLflow UI Anda berjalan di http://127.0.0.1:5001 dan model/versi yang ditentukan ada.")
