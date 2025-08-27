import os
import pickle

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_model(category: str):
    model_path = os.path.join("models", f"{category}_model.pkl")
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):  # saved as {"model": ..., "transformer": ...}
        return obj["model"], obj.get("transformer", None)

    return obj, None  # saved as raw model
