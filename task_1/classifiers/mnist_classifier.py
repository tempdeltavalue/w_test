from task_1.classifiers.cnn_classifier import CNNClassifier, NNClassifier, RFClassifier

from sklearn.metrics import accuracy_score

import torch

class MnistClassifier:
    # Central place for configuration
    def __init__(self, algorithm: str, train_data, test_data_full, 
                 output_dir, device, 
                 num_epochs, patience, 
                 learning_rate, batch_size):
        
        self.algorithm = algorithm.lower()
        self.output_dir = output_dir
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Store hyperparameters
        self.num_epochs = num_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.classifier_instance = self._initialize_classifier()
        
        # Note: load_mnist_data now needs batch_size, handle this outside of __init__ or use a factory
        # For simplicity here, we assume data is already loaded using the correct batch_size
        self.classifier_instance.train(train_data)
        self.X_test_numpy, self.y_test_numpy = test_data_full

    def _initialize_classifier(self):
        print(f"\nInitializing classifier: {self.algorithm.upper()}")
        
        # Parameters to pass to specific classifiers (including new hyperparameters)
        kwargs = {
            'output_dir': self.output_dir, 
            'device': self.device,
            'num_epochs': self.num_epochs,
            'patience': self.patience,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        
        # Pass the central configuration down to the specific classifier
        if self.algorithm == 'rf':
            return RFClassifier(**kwargs)
        elif self.algorithm == 'nn':
            return NNClassifier(**kwargs)
        elif self.algorithm == 'cnn':
            return CNNClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Choose 'rf', 'nn', or 'cnn'.")

    def predict(self):
        if self.algorithm == 'rf':
            X_test_features = self.X_test_numpy
        else:
            X_test_features = self.X_test_numpy.reshape(-1, 28, 28)
            
        predictions = self.classifier_instance.predict(X_test_features)
        
        return predictions, self.y_test_numpy

    def evaluate(self, predictions, true_labels):
        accuracy = accuracy_score(true_labels, predictions)
        print(f"\n--- {self.algorithm.upper()} Evaluation ---")
        print(f"Final Accuracy: {accuracy:.4f} ({(accuracy * 100):.2f}%)")
        
        history = self.classifier_instance.history
        if history:
            print("\nHistory:")
            for record in history:
                train_loss_str = f"Train Loss={record['loss']:.4f} | " if record['loss'] != 0.0 else ""
                val_loss_str = f"Val Loss={record['val_loss']:.4f} | " if record['val_loss'] != 0.0 else ""
                
                print(f"  Epoch {record['epoch']}: {train_loss_str}{val_loss_str}Acc={record['accuracy']:.4f} | F1={record['f1_score']:.4f}")
        print("--------------------------------")