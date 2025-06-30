from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
import pickle

'''
class ModelManager:
    def __init__(self, storage_dir: str, max_loaded_models: int):
        self.storage_dir = storage_dir
        self.max_loaded_models = max_loaded_models
        self.loaded_models: Dict[str, Any] = {}
        self.lock = Lock()

        os.makedirs(self.storage_dir, exist_ok=True)

    def _get_model_path(self, name: str) -> str:
        return os.path.join(self.storage_dir, f"{name}.joblib")

    def _create_model(self, model_type: str, config: dict = {}) -> Any:
        model_type = model_type.lower()
        if model_type == "logreg":
            return LogisticRegression(**config)
        elif model_type == "linreg":
            return LinearRegression(**config)
        elif model_type == "kneighbors":
            return KNeighborsClassifier(**config)
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

    def fit(self, name: str, X, y, model_type: str = "logreg", config: dict = {}):
        with self.lock:
            if name in self.loaded_models:
                raise ValueError(f"Модель '{name}' уже загружена")

            model = self._create_model(model_type, config)
            model.fit(X, y)

            path = self._get_model_path(name)
            joblib.dump(model, path)

    def load(self, name: str):
        with self.lock:
            if name in self.loaded_models:
                raise ValueError(f"Модель '{name}' уже загружена")
            if len(self.loaded_models) >= self.max_loaded_models:
                raise RuntimeError("Превышено число одновременно загруженных моделей")

            path = self._get_model_path(name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл модели '{name}' не найден")

            model = joblib.load(path)
            self.loaded_models[name] = model

    def unload(self, name: str):
        with self.lock:
            if name not in self.loaded_models:
                raise ValueError(f"Модель '{name}' не загружена")
            del self.loaded_models[name]

    def predict(self, name: str, X):
        with self.lock:
            if name not in self.loaded_models:
                raise ValueError(f"Модель '{name}' не загружена")
            preds = self.loaded_models[name].predict(X)
            return preds.tolist() if hasattr(preds, 'tolist') else preds

    def remove(self, name: str):
        with self.lock:
            if name in self.loaded_models:
                del self.loaded_models[name]
            path = self._get_model_path(name)
            if os.path.exists(path):
                os.remove(path)
            else:
                raise FileNotFoundError(f"Файл модели '{name}' не найден")

    def remove_all(self):
        with self.lock:
            self.loaded_models.clear()
            for fname in os.listdir(self.storage_dir):
                if fname.endswith(".joblib"):
                    os.remove(os.path.join(self.storage_dir, fname))
'''

class ModelManager:
    def __init__(self, model_type: str, **params):
        self.model_type = model_type
        if model_type == "logreg":
            self.model = LogisticRegression(**params)
        elif model_type == "linreg":
            self.model = LinearRegression(**params)
        # elif model_type == "knn":
        #     self.model = KNeighborsClassifier(**params)
        elif model_type == "RandomForest": 
            self.model =  RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown model type '{model_type}'") 

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.model_type, self.model), f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            model_type, model = pickle.load(f)
        wrapper = cls(model_type)
        wrapper.model = model
        return wrapper
