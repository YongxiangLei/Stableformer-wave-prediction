import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import importlib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os
from data_transform_multi import load_csv_files, normalize_data, create_timestamps_ds, split_dataset  # Import necessary functions

def load_model_class(model_name, model_category):
    module_name = f"{model_category}.{model_name}"
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, 'Model')  # 'Model' is the class name in Transformer.py
        return model_class
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Model {model_name} not found in {model_category}")

def select_model(model_name, model_category, configs):
    model_class = load_model_class(model_name, model_category)
    return model_class(configs)

def train_model(model_name, model_category, train_loader, device, configs, epochs=50):
    model = select_model(model_name, model_category, configs).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            dec_inp = batch_features[:, -configs.pred_len:, :]  # Prepare the decoder input for forecasting
            outputs = model(batch_features, None, dec_inp, None)  # Assuming no time encoding

            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model

def evaluate_model(model, test_loader, device, configs):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            dec_inp = batch_features[:, -configs.pred_len:, :]
            outputs = model(batch_features, None, dec_inp, None)

            predictions.append(outputs.cpu().numpy())
            targets.append(batch_labels.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    mse = mean_squared_error(targets, predictions)
    print(f"Test MSE: {mse:.4f}")

    return predictions, targets

def plot_results(predictions, targets, title="Model Results", save_path=None, sample_idx=None):
    """
    Plots the true and predicted values for each test sample over the prediction horizon.

    Args:
        predictions (ndarray): The predicted values (shape: [num_samples, pred_len, 1]).
        targets (ndarray): The true values (shape: [num_samples, pred_len, 1]).
        title (str): Title for the plot.
        save_path (str): If provided, the plot will be saved to this path.
        sample_idx (int): Index of the sample to plot. If None, plots all samples.
    """
    num_samples = predictions.shape[0]
    if sample_idx is not None:
        sample_indices = [sample_idx]  # Single sample plot
    else:
        sample_indices = range(num_samples)  # Plot all samples

    for idx in sample_indices:
        pred_sample = predictions[idx].flatten()  # Shape: (pred_len,)
        true_sample = targets[idx].flatten()      # Shape: (pred_len,)

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(true_sample)), true_sample, label='True', marker='o', color='blue')
        plt.plot(range(len(pred_sample)), pred_sample, label='Predicted', marker='x', color='red')

        plt.title(f"{title} - Sample {idx + 1}")
        plt.xlabel("Prediction Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        if save_path:
            sample_save_path = save_path.replace(".png", f"_sample_{idx+1}.png")
            plt.savefig(sample_save_path)
        plt.show()

def main():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_folder = "Cases"
    batch_size = 4
    epochs = 5

    models_to_test = ['Transformer_v1']
    model_category = 'base_models'  # Use 'diffused_models' or 'prob_models' for other categories

    # Transformer specific configurations
    class Configs:
        enc_in = 1           # Input size
        dec_in = 1           # Decoder input size
        c_out = 1            # Output size
        d_model = 64         # Model dimension
        embed = 'timeF'      # Embedding type
        freq = 'h'           # Frequency
        dropout = 0.1        # Dropout rate
        factor = 5           # Attention factor
        n_heads = 8          # Number of attention heads
        d_ff = 128           # Feedforward layer dimension
        e_layers = 2         # Number of encoder layers
        d_layers = 1         # Number of decoder layers
        activation = 'gelu'  # Activation function
        learning_rate = 0.0001
        his_len = 300        # Historical length
        pred_len = 100        # Prediction length (multi-step)
    
    configs = Configs()

    # Load all CSV files and process them
    all_csv_data = load_csv_files(data_folder)

    for i, csv_data in enumerate(all_csv_data):
        # Normalize data
        normalized_data = normalize_data(csv_data)

        # Create time series dataset
        features, labels = create_timestamps_ds(normalized_data.flatten(), history_size=configs.his_len, prediction_size=configs.pred_len)

        # Split dataset into training and testing
        train_features, test_features, train_labels, test_labels = split_dataset(features, labels, test_size=0.1)

        # Reshape the features and labels for model input
        train_features = train_features.unsqueeze(-1)  # Change shape to (num_samples, sequence_length, 1)
        test_features = test_features.unsqueeze(-1)    # Change shape to (num_samples, sequence_length, 1)

        train_labels = train_labels.unsqueeze(-1)      # Change shape to (num_samples, pred_len, 1)
        test_labels = test_labels.unsqueeze(-1)        # Change shape to (num_samples, pred_len, 1)

        # Output the shapes of the datasets
        print(f"File SS{i + 1} Train features shape: {train_features.shape}, Train labels shape: {train_labels.shape}")
        print(f"File SS{i + 1} Test features shape: {test_features.shape}, Test labels shape: {test_labels.shape}\n")

        # Create DataLoader for batch processing
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for model_name in models_to_test:
            model_save_path = os.path.join("models", f"{model_name}_SS{i + 1}.pth")
            if os.path.exists(model_save_path):
                print(f"Loading pre-trained model {model_name} for file SS{i + 1}...")
                model = select_model(model_name, model_category, configs).to(device)
                model.load_state_dict(torch.load(model_save_path))
            else:
                print(f"Training {model_name} on file SS{i + 1}...")
                model = train_model(model_name, model_category, train_loader, device, configs, epochs=epochs)
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved model to {model_save_path}")

            # Evaluate the model
            predictions, targets = evaluate_model(model, test_loader, device, configs)

            # Save plots for all samples
            plot_save_path = os.path.join("plots", f"{model_name}_Results_SS{i + 1}.png")
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
            plot_results(predictions, targets, title=f"{model_name} Results on file SS{i + 1}", save_path=plot_save_path, sample_idx=None)  # Plot all samples
            print(f"Saved plots for all test samples to {plot_save_path}")

if __name__ == "__main__":
    main()
