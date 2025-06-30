Структура проекта:

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


для запуска (я работал в WSL:UBUNTU) нужно перейти в корень проекта и выполнить:

1. Построение сервера:

docker build -t ml-server ./server

2. Запуск сервера :

docker run \
  -p 8991:8991 \
  -v "$(pwd)/models:/ml_server/models" \
  -v /tmp:/tmp \
  --env-file server/.env \
  --name ml-server-debug \
  ml-server

Модели сохраняются в директорию models/.

Данные для обучения могут временно сохраняться в /tmp в формате .npy - но это для асинхронного обучения.

Порт 8991 пробрасывается наружу.

В директории client/ доступны 2 файла:

1. client.ipynb
Jupyter Notebook с функциями:

client_fit(X, y, model_type, config) — запуск обучения модели (в том же процессе).

client_predict(name, X) — получить предсказания.

client_load(name) / client_unload(name) — загрузка/выгрузка модели в память.

client_remove(name) / client_remove_all() — удаление одной или всех моделей.

2. client_async.py
Скрипт для асинхронного обучения нескольких моделей:

Его запуск прописан в client.ipynb

Поддерживаемые модели:
logreg – логистическая регрессия
linreg – линейная регрессия
RandomForest.