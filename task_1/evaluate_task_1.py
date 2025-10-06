
import torch
from task_1.classifiers.mnist_classifier import MnistClassifier
from task_1.utils import load_mnist_data
# Define defaults centrally, but they will be passed to classes via MnistClassifier

DEFAULT_NUM_EPOCHS = 20
DEFAULT_PATIENCE = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001




if __name__ == "__main__":
    # Configure environment and hyperparameters globally here
    custom_output_dir = './mnist_results_final_hparams'
    custom_device = torch.device("cpu") # For demonstration, forcing CPU
    custom_lr = 0.005 # Example: higher learning rate for NN/CNN
    custom_epochs = 10 # Example: shorter training for all
    
    print(f"--- Global Configuration ---")
    print(f"Output Directory: {custom_output_dir}")
    print(f"Device: {custom_device}")
    print(f"Learning Rate: {custom_lr}")
    print(f"Epochs: {custom_epochs}")
    print("--- Global Data Loading ---")
    
    # Load data using the default batch size (64) or a custom one if defined
    (train_loader, test_loader_nn_cnn, (X_train_rf, y_train_rf, X_test_rf, y_test_rf), X_test_nn_cnn) = load_mnist_data(batch_size=DEFAULT_BATCH_SIZE)

    rf_data = (X_train_rf, y_train_rf, X_test_rf, y_test_rf)
    nn_cnn_data = (train_loader, test_loader_nn_cnn)

    test_data_full = (X_test_rf, y_test_rf)

    # Random Forest
    rf_runner = MnistClassifier(
        algorithm='rf', 
        train_data=rf_data, 
        test_data_full=test_data_full,
        output_dir=custom_output_dir,
        device=custom_device,
        num_epochs=custom_epochs # Passes this param, though RF ignores it
    )
    rf_preds, rf_true = rf_runner.predict()
    rf_runner.evaluate(rf_preds, rf_true)

    # Feed-Forward Neural Network
    nn_runner = MnistClassifier(
        algorithm='nn', 
        train_data=nn_cnn_data, 
        test_data_full=test_data_full,
        output_dir=custom_output_dir,
        device=custom_device,
        num_epochs=custom_epochs, 
        learning_rate=custom_lr
    )
    nn_preds, nn_true = nn_runner.predict()
    nn_runner.evaluate(nn_preds, nn_true)

    # Convolutional Neural Network
    cnn_runner = MnistClassifier(
        algorithm='cnn', 
        train_data=nn_cnn_data, 
        test_data_full=test_data_full,
        output_dir=custom_output_dir,
        device=custom_device,
        num_epochs=custom_epochs,
        learning_rate=custom_lr
    )
    cnn_preds, cnn_true = cnn_runner.predict()
    cnn_runner.evaluate(cnn_preds, cnn_true)