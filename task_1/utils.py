
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from task_1.mnist_array_dataset import MnistArrayDataset

def load_mnist_data(BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    rf_train_raw = datasets.MNIST('./data', train=True, download=True)
    rf_test_raw = datasets.MNIST('./data', train=False, download=True)

    X_train_rf = rf_train_raw.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_train_rf = rf_train_raw.targets.numpy()
    X_test_rf = rf_test_raw.data.numpy().reshape(-1, 28 * 28) / 255.0
    y_test_rf = rf_test_raw.targets.numpy()

    X_test_nn_cnn = rf_test_raw.data.numpy() / 255.0
    test_array_dataset = MnistArrayDataset(X_test_nn_cnn, y_test_rf)

    return (train_loader, test_loader, test_array_dataset, 
            (X_train_rf, y_train_rf, X_test_rf, y_test_rf))
