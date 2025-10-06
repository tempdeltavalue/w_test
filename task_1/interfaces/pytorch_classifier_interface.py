import torch.nn as nn
import torch.optim as optim

from abc import abstractmethod

from task_1.interfaces.mnist_classifier_interface import MnistClassifierInterface

class PyTorchClassifierInterface(MnistClassifierInterface):
    
    @abstractmethod
    def _evaluate_epoch(self, test_loader):
        pass
    
    # Unified PyTorch training loop
    def train(self, data):
        # We assume data is (train_loader, test_loader_nn_cnn)
        train_loader, test_loader_nn_cnn = data
        
        if self._load_model():
            return
            
        print(f"    [{self.__class__.__name__[:3]}] Training PyTorch Network on {self.device}...")
        
        criterion = nn.CrossEntropyLoss()
        # Use self.learning_rate here
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_acc = 0.0
        epochs_no_improve = 0
        best_model_state = self.model.state_dict()

        self.model.train()
        # Use self.num_epochs here
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            
            # Training Step
            for data, targets in train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            
            # Evaluation Step (calls subclass implementation)
            val_metrics = self._evaluate_epoch(test_loader_nn_cnn)
            
            self._record_epoch_history(epoch, epoch_loss, val_metrics)
            
            epoch_acc = val_metrics[1]
            print(f"    [{self.__class__.__name__[:3]}] Epoch {epoch+1}/{self.num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_metrics[0]:.4f} | Acc: {epoch_acc:.4f} | F1: {val_metrics[4]:.4f}")

            # Early Stopping Logic
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                epochs_no_improve = 0
                best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1
            
            # Use self.patience here
            if epochs_no_improve == self.patience:
                print(f"    [{self.__class__.__name__[:3]}] Early stopping triggered after {epoch + 1} epochs. No improvement for {self.patience} epochs.")
                break

        self.model.load_state_dict(best_model_state)
        print(f"    [{self.__class__.__name__[:3]}] Restored model with best accuracy: {best_acc:.4f}")

        self._save_model()