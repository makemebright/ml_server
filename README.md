# ML_SERVER

This document describes the structure and usage of the ML_SERVER project.

## Project Structure
```bash
ML_SERVER/
│
├── client/
│   ├── client_async.py     # Скрипт для асинхронного обучения моделей
│   └── client.ipynb        # Jupyter-ноутбук для тестов и демонстрации работы
│
├── models/                 # Сохранённые модели после обучения (остатки, можно удалить их и рабоать с новыми, а можно оставить)
│   ├── logreg.pkl
│   ├── logreg_1.pkl
│   ├── linreg.pkl
│   └── linreg_1.pkl
│
├── server/
│   ├── classifier.py       # Классы моделей и логика обучения/сохранения/предсказаний
│   ├── server.py           # Основной FastAPI сервер
│   ├── Dockerfile          # Docker-образ для запуска сервера
│   ├── requirements.txt    # Зависимости Python
│   └── .env                # Настройки сервера (.env-файл)
│
└── README.md               # Этот файл
```

## How to Run (WSL:UBUNTU environment)

Navigate to the project root and execute the following commands:

1.  **Build the server:**

    ```bash
    docker build -t ml-server ./server
    ```

2.  **Run the server:**

    ```bash
    docker run \
      -p 8991:8991 \
      -v "$(pwd)/models:/ml_server/models" \
      -v /tmp:/tmp \
      --env-file server/.env \
      --name ml-server-debug \
      ml-server
    ```

    * Models are saved to the `models/` directory.
    * Training data for asynchronous training may be temporarily saved in `/tmp` in `.npy` format.
    * Port 8991 is exposed.

## Client Directory

The `client/` directory contains two files:

1.  **`client.ipynb`**
    Jupyter Notebook with the following functions:
    * `client_fit(X, y, model_type, config)` — initiates model training (in the same process).
    * `client_predict(name, X)` — retrieves predictions.
    * `client_load(name) / client_unload(name)` — loads/unloads a model into/from memory.
    * `client_remove(name) / client_remove_all()` — deletes one or all models.

2.  **`client_async.py`**
    A script for asynchronous training of multiple models. Its execution is defined in `client.ipynb`.

## Supported Models

* `logreg` – Logistic Regression
* `linreg` – Linear Regression
* `RandomForest`
