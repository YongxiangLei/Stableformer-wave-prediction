import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import importlib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os
from data_transform import load_csv_files, normalize_data, create_timestamps_ds, split_dataset  # Import necessary functions

def load_model_class(model_name, model_category):
    module_name = f"{model_category}.{model_name}"
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, model_name)
        return model_class
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Model {model_name} not found in {model_category}")

def select_model(model_name, model_category):
    return load_model_class(model_name, model_category)

def train_and_evaluate(model_name, model_category, train_loader, test_loader, device, epochs=50):
    model_class = select_model(model_name, model_category)
    model = model_class(input_size=1, hidden_size=20, num_layers=2, output_size=1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluate on test data
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_labels.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    mse = mean_squared_error(targets, predictions)
    print(f"Test MSE: {mse:.4f}")

    return predictions, targets, model

def plot_results(predictions, targets, title="Model Results", save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(targets, label='True')
    plt.plot(predictions, label='Predicted')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path,dpi=1200)
    #plt.show()

if __name__ == "__main__":
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_folder = "Cases"
    batch_size = 32
    epochs = 50

    models_to_test = ['Transformer']
    # models_to_test = ['NN','LSTM', 'GRU', 'TCN', 'RNN','Transformer']
    model_category = 'base_models'  # Use 'diffused_models' or 'prob_models' for other categories

    # Load all CSV files and process them
    all_csv_data = load_csv_files(data_folder)

    for i, csv_data in enumerate(all_csv_data):
        # Normalize data
        normalized_data = normalize_data(csv_data)

        # Create time series dataset
        features, labels = create_timestamps_ds(normalized_data.flatten())

        # Split dataset into training and testing
        train_features, test_features, train_labels, test_labels = split_dataset(features, labels, test_size=0.2)

        # Reshape the features and labels for model input
        train_features = train_features.unsqueeze(-1)  # Change shape to (num_samples, sequence_length, 1)
        test_features = test_features.unsqueeze(-1)    # Change shape to (num_samples, sequence_length, 1)

        train_labels = train_labels.unsqueeze(-1)      # Change shape to (num_samples, 1)
        test_labels = test_labels.unsqueeze(-1)        # Change shape to (num_samples, 1)

        # Output the shapes of the datasets
        print(f"文件 SS{i + 1} 的训练集特征 shape: {train_features.shape}, 训练集标签 shape: {train_labels.shape}")
        print(f"文件 SS{i + 1} 的测试集特征 shape: {test_features.shape}, 测试集标签 shape: {test_labels.shape}\n")

        # Create DataLoader for batch processing
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for model_name in models_to_test:
            print(f"Training and evaluating {model_name} on file SS{i + 1}...")

            predictions, targets, model = train_and_evaluate(model_name, model_category, train_loader, test_loader, device, epochs=epochs)

            # Save model
            model_save_path = os.path.join("models", f"{model_name}_SS{i + 1}.pth")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved model to {model_save_path}")

            # Save plot
            plot_save_path = os.path.join("plots", f"{model_name}_Results_SS{i + 1}.png")
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
            plot_results(predictions, targets, title=f"{model_name} Results on file SS{i + 1}", save_path=plot_save_path)
            print(f"Saved plot to {plot_save_path}")
