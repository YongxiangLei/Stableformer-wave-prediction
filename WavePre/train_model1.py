import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import importlib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from data_transform_multi import load_csv_files, normalize_data, create_timestamps_ds, split_dataset 

# Create directory for saving logs, models, and figures
output_dir = "./training_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

def train_and_evaluate(model_name, model_category, train_loader, test_loader, device, epochs=50, log_file=None):
    model_class = select_model(model_name, model_category)
    model = model_class(input_size=1, hidden_size=20, num_layers=2, output_size=20).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize lists to store logs
    train_losses = []
    test_mses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Logging the training loss
        if log_file:
            log_file.write(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}\n")
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Evaluate on test data at the end of every epoch
        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)

                # Ensure that you append the correct number of samples
                predictions.append(outputs.cpu().numpy())
                targets.append(batch_labels.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)[:len(test_loader.dataset)]  # Match the length of the test set
        targets = np.concatenate(targets, axis=0)[:len(test_loader.dataset)]  # Match the length of the test set
        mse = mean_squared_error(targets, predictions)
        test_mses.append(mse)

        if log_file:
            log_file.write(f"Epoch [{epoch+1}/{epochs}], Test MSE: {mse:.4f}\n")
        print(f"Test MSE after Epoch [{epoch+1}/{epochs}]: {mse:.4f}")

    return model, train_losses, test_mses, predictions, targets


def plot_and_save_results(predictions, targets, model_name, file_num, train_losses, test_mses):
    # Create figure for prediction results
    plt.figure()
    plt.plot(targets.flatten(), label='True')
    plt.plot(predictions.flatten(), label='Predicted')
    plt.title(f"{model_name} Results on file SS{file_num}")
    plt.legend()
    
    # Save figure to file
    figure_path = os.path.join(output_dir, f"{model_name}_SS{file_num}_results.png")
    plt.savefig(figure_path)
    plt.close()

    # Plot training loss and test MSE
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_mses, label='Test MSE')
    plt.title(f"Training Loss and Test MSE for {model_name} on file SS{file_num}")
    plt.legend()
    
    # Save loss and MSE plot
    loss_mse_path = os.path.join(output_dir, f"{model_name}_SS{file_num}_loss_mse.png")
    plt.savefig(loss_mse_path)
    plt.close()

def save_model(model, model_name, file_num):
    # Save model
    model_save_path = os.path.join(output_dir, f"{model_name}_SS{file_num}.pth")
    torch.save(model.state_dict(), model_save_path)

def log_training_start(log_file, model_name, file_num):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file.write(f"\n{'-'*50}\n")
    log_file.write(f"Training started for {model_name} on file SS{file_num} at {now}\n")
    log_file.write(f"{'-'*50}\n")

if __name__ == "__main__":
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_folder = "Cases"
    batch_size = 32
    epochs = 50

    
    models_to_test = ['LSTM', 'GRU', 'TCN', 'RNN', 'NN', 'Transformer']
    model_category = 'base_models'  # Use 'diffused_models' or 'prob_models' for other categories

    # Load all CSV files and process them
    all_csv_data = load_csv_files(data_folder)

    # Create log file
    log_file_path = os.path.join(output_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:

        for i, csv_data in enumerate(all_csv_data):
            # Normalize data
            normalized_data = normalize_data(csv_data)

            # Create multi-step time series dataset
            features, labels = create_timestamps_ds(normalized_data.flatten(), history_size=100, prediction_size=20)

            # Split dataset into training and testing
            train_features, test_features, train_labels, test_labels = split_dataset(features, labels, test_size=0.2)
            # Reshape the features and labels for model input
            train_features = train_features.unsqueeze(-1)  # Shape: (num_samples, history_size, 1)
            test_features = test_features.unsqueeze(-1)    # Shape: (num_samples, history_size, 1)

            train_labels = train_labels  # Shape: (num_samples, prediction_size)
            test_labels = test_labels    # Shape: (num_samples, prediction_size)

            # Output the shapes of the datasets
            print(f"文件 SS{i + 1} 的训练集特征 shape: {train_features.shape}, 训练集标签 shape: {train_labels.shape}")
            print(f"文件 SS{i + 1} 的测试集特征 shape: {test_features.shape}, 测试集标签 shape: {test_labels.shape}\n")

            # Create DataLoader for batch processing
            train_dataset = TensorDataset(train_features, train_labels)
            test_dataset = TensorDataset(test_features, test_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            for model_name in models_to_test:
                log_training_start(log_file, model_name, i + 1)

                # Train and evaluate the model
                print(f"Training and evaluating {model_name} on file SS{i + 1}...")
                model, train_losses, test_mses, predictions, targets = train_and_evaluate(
                    model_name, model_category, train_loader, test_loader, device, epochs=epochs, log_file=log_file
                )

                # Save model, plots, and results
                save_model(model, model_name, i + 1)
                plot_and_save_results(predictions, targets, model_name, i + 1, train_losses, test_mses)
                log_file.write(f"{model_name} on file SS{i + 1} completed.\n\n")