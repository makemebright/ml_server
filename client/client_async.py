import threading
import time
import requests
import numpy as np
import uuid
import os

MY_URL = "http://localhost:8991"

def generate_classification_dataset(n_samples=400000, n_features=100):
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, size=n_samples)
    return X, y

def generate_regression_dataset(n_samples=400000, n_features=100):
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples)
    return X, y

def save_numpy_array(arr, name):
    TMP_DIR = "/tmp"
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    path = os.path.join(TMP_DIR, f"{name}_.npy")
    np.save(path, arr)
    return path

def client_fit_from_file(X_path, y_path, model_type="logreg", config={}):
    try:
        response = requests.post(f"{MY_URL}/fit_from_file", json={
            "model_type": model_type,
            "config": config,
            "X_path": X_path,
            "y_path": y_path
        })
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"[client_fit_from_file] Ошибка HTTP: {response.text}")
    except Exception as e:
        print(f"[client_fit_from_file] Ошибка: {e}")

def train_async(name, X_path, y_path):
    print(f"[{name}] Запуск обучения...")
    resp = client_fit_from_file(X_path, y_path, model_type=name)
    print(f"[{name}] Запрос на обучение отправлен.")

if __name__ == "__main__":
    X_log, y_log = generate_classification_dataset()
    X_lin, y_lin = generate_regression_dataset()

    X_log_path = save_numpy_array(X_log, "X_log")
    y_log_path = save_numpy_array(y_log, "y_log")
    X_lin_path = save_numpy_array(X_lin, "X_lin")
    y_lin_path = save_numpy_array(y_lin, "y_lin")

    t1 = threading.Thread(target=train_async, args=("logreg", X_log_path, y_log_path))
    t2 = threading.Thread(target=train_async, args=("linreg", X_lin_path, y_lin_path))

    overall_start = time.time()

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    overall_duration = time.time() - overall_start

    print("Ждем завершения обучения моделей (примерно 90 секунд), а затем смотрб время в логах")
    time.sleep(90)

    print(f"\n[ИТОГО] Клиент завершил работу за {overall_duration:.2f} секунд")