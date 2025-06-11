import mlflow
import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

def train_and_log_model():
    print("Memulai proses pelatihan model...")

    data_path = "ecommerce_shipping_data_preprocessed"
    if not os.path.exists(data_path):
        print(f"Error: Direktori data '{data_path}' tidak ditemukan.")
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

    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True
    )

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
    print(f"Akurasi model: {acc}")

    print("Logging model explicitly to artifact store...")
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model" # Ini yang terpenting: menyimpan artefak ke folder 'model'
    )
    print("Model explicitly logged successfully.")


if __name__ == "__main__":
    train_and_log_model()