import torch
from abc import ABC, abstractmethod
from joblib import dump, load

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MnistClassifierInterface(ABC):
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            print(f"    Loading model from {self.model_path}...")
            if 'rf' in self.model_path:
                self.model = load(self.model_path)
            else:
                self.model = self._initialize_model()
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.to(DEVICE)
                self.model.eval()
            return True
        return False

    @abstractmethod
    def _initialize_model(self):
        pass
    
    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def _save_model(self):
        print(f"    Saving model to {self.model_path}...")
        if 'rf' in self.model_path:
            dump(self.model, self.model_path)
        else:
            torch.save(self.model.state_dict(), self.model_path)