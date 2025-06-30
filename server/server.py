from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import pickle
import multiprocessing
import traceback
import uuid
import classifier
from dotenv import load_dotenv
import uvicorn
import threading
from datetime import datetime

load_dotenv(dotenv_path=".env")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
N_CORES = int(os.getenv("N_CORES", "2"))
MAX_LOADED = int(os.getenv("MAX_LOADED", "1"))

os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI()
loaded_models = {}
loaded_models_lock = threading.Lock()

active_train_processes = []
active_train_lock = threading.Lock()

class FitRequest(BaseModel):
    model_type: str
    config: dict = {}
    X: list
    y: list

class FitFromFileRequest(BaseModel):
    model_type: str
    config: dict = {}
    X_path: str
    y_path: str

class PredictRequest(BaseModel):
    name: str
    X: list

def uniqie_name(model_type: str) -> str:
    base_name = model_type
    i = 0
    while True:
        if i == 0:
            candidate = f"{base_name}.pkl"
        else:
            candidate = f"{base_name}_{i}.pkl"
        if not os.path.exists(os.path.join(MODEL_DIR, candidate)):
            return os.path.splitext(candidate)[0]
        i += 1

def train_and_save_model(path, model_type, config, X, y):
    try:
        print(f"Начало обучения {path} в {datetime.now()}") # это тоже костыль для асинхронного обучения
        model = classifier.ModelManager(model_type, **config)
        model.fit(X, y)
        model.save(path)
        print(f"Модель сохранена в {path}")
        print(f"Обучение модели {path} завершено в {datetime.now()}") # это костыль для асинхронного обучения
    except Exception as e:
        print(f"Ошибка : {e}")
        traceback.print_exc()

@app.get("/")
def root():
    return {"message": "ML server is running"}

@app.post("/fit")
def fit(request: FitRequest):
    X = np.array(request.X)
    y = np.array(request.y)

    model_name = uniqie_name(request.model_type)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    with active_train_lock:
        active_train_processes[:] = [p for p in active_train_processes if p.is_alive()]
        if len(active_train_processes) >= N_CORES - 1:
            raise HTTPException(status_code=400, detail="Нет доступных ядер для обучения модели")

        p = multiprocessing.Process(
            target=train_and_save_model,
            args=(model_path, request.model_type, request.config, X, y)
        )
        p.start()
        active_train_processes.append(p)

    return {"message": f"Обучение модели '{model_name}' запущено", "pid": p.pid}

@app.post("/fit_from_file")
def fit_from_file(request: FitFromFileRequest):
    try:
        X = np.load(request.X_path)
        y = np.load(request.y_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки файлов: {e}")

    model_name = uniqie_name(request.model_type)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    with active_train_lock:
        active_train_processes[:] = [p for p in active_train_processes if p.is_alive()]
        if len(active_train_processes) >= N_CORES - 1:
            raise HTTPException(status_code=400, detail="Нет доступных ядер для обучения модели")

        p = multiprocessing.Process(
            target=train_and_save_model,
            args=(model_path, request.model_type, request.config, X, y)
        )
        p.start()
        active_train_processes.append(p)

    return {"message": f"Обучение модели '{model_name}' запущено", "pid": p.pid}

@app.post("/load")
def load(name: str):
    with loaded_models_lock:
        if name in loaded_models:
            raise HTTPException(status_code=400, detail=f"Модель '{name}' уже загружена")
        if len(loaded_models) >= MAX_LOADED:
            raise HTTPException(status_code=400, detail="Превышен лимит одновременно загруженных моделей")

        model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail=f"Модель '{name}' не найдена")

        try:
            loaded_models[name] = classifier.ModelManager.load(model_path)
            return {"message": f"Модель '{name}' загружена"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/unload")
def unload(name: str):
    with loaded_models_lock:
        if name in loaded_models:
            del loaded_models[name]
            return {"message": f"Модель '{name}' выгружена"}
        else:
            raise HTTPException(status_code=400, detail=f"Модель '{name}' не загружена")

@app.post("/predict")
def predict(request: PredictRequest):
    X = np.array(request.X)
    with loaded_models_lock:
        if request.name not in loaded_models:
            raise HTTPException(status_code=400, detail=f"Модель '{request.name}' не загружена")
        try:
            preds = loaded_models[request.name].predict(X).tolist()
            return {"predictions": preds}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/remove")
def remove(name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    try:
        with loaded_models_lock:
            if name in loaded_models:
                del loaded_models[name]
        if os.path.exists(path):
            os.remove(path)
        else:
            raise HTTPException(status_code=400, detail="Файл модели не найден")
        return {"message": f"Модель '{name}' удалена"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/remove_all")
def remove_all():
    try:
        with loaded_models_lock:
            loaded_models.clear()
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".pkl"):
                os.remove(os.path.join(MODEL_DIR, f))
        return {"message": "ВСе модели удалены"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    print("Running")
    start_time = datetime.now()
    uvicorn.run(app, host="0.0.0.0", port=8991)