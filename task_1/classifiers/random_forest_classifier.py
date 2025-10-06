from sklearn.ensemble import RandomForestClassifier
from task_1.classifiers.mnist_classifier_interface import MnistClassifierInterface

class RFClassifier(MnistClassifierInterface):
    def __init__(self):
        super().__init__(os.path.join(OUTPUT_DIR, 'rf_model.joblib'))

    def _initialize_model(self):
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    def train(self, data):
        X_train, y_train, _, _ = data
        
        if self.model is None:
            self.model = self._initialize_model()
        
        if self._load_model():
             return

        print("    [RF] Training Random Forest (100 trees)...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        print(f"    [RF] Training finished in {end_time - start_time:.2f} seconds.")
        
        self._save_model()

    def predict(self, X_test):
        X_test_flat = X_test.reshape(-1, 28 * 28)
        return self.model.predict(X_test_flat)