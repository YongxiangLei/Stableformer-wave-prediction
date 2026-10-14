import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import importlib
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from data_transform import load_csv_files, normalize_data, create_timestamps_ds, split_dataset  # Import necessary functions

# Load model class from the provided module
def load_model_class(model_name, model_category):
    module_name = f"{model_category}.{model_name}"
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, model_name)
        return model_class
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Model {model_name} not found in {model_category}")

# Select the model class
def select_model(model_name, model_category):
    return load_model_class(model_name, model_category)

# Custom loss function to compute the log likelihood for a Gaussian distribution
def gaussian_nll(mu, sigma, y):
    dist_normal = torch.distributions.Normal(mu, sigma)
    log_likelihood = dist_normal.log_prob(y)
    return -log_likelihood.mean()

# Training and evaluation function for probabilistic models
def train_and_evaluate(model_name, model_category, train_loader, test_loader, device, epochs=50):
    model_class = select_model(model_name, model_category)
    model = model_class(input_size=1, hidden_size=20, num_layers=2, output_size=1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            # Forward pass
            mu, sigma = model(batch_features)
            loss = gaussian_nll(mu, sigma, batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluate on test data
    model.eval()
    predictions = []
    targets = []
    sigmas = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            mu, sigma = model(batch_features)
            predictions.append(mu.cpu().numpy())
            targets.append(batch_labels.cpu().numpy())
            sigmas.append(sigma.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    sigmas = np.concatenate(sigmas, axis=0)

    mse = mean_squared_error(targets, predictions)
    print(f"Test MSE: {mse:.4f}")

    return predictions, targets, sigmas, model

# Plot results with confidence intervals
def plot_results(predictions, targets, sigmas, title="Model Results", save_path=None):
    plt.figure(figsize=(10, 5))

    predictions = predictions.flatten()
    targets = targets.flatten()
    sigmas = sigmas.flatten()
    
    # Calculate upper and lower confidence bounds using 1.96 * sigma (for ~95% confidence interval)
    upper_bound = predictions + 1.96 * sigmas
    lower_bound = predictions - 1.96 * sigmas
    
    plt.plot(targets, label='True', color='blue')
    plt.plot(predictions, label='Predicted', color='red')
    
    # Shaded confidence interval
    plt.fill_between(range(len(predictions)), lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')
    
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    #plt.show()

if __name__ == "__main__":
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_folder = "Cases"
    batch_size = 32
    epochs = 50

    models_to_test = ['Transformer_Prob','NN_Prob','GRU_Prob','RNN_Prob','TCN_Prob']  # Add other probabilistic models if needed
    model_category = 'prob_models'

    # Load and process CSV files
    all_csv_data = load_csv_files(data_folder)

    for i, csv_data in enumerate(all_csv_data):
        # Normalize data
        normalized_data = normalize_data(csv_data)

        # Create time series dataset
        features, labels = create_timestamps_ds(normalized_data.flatten())

        # Split dataset into training and testing
        train_features, test_features, train_labels, test_labels = split_dataset(features, labels, test_size=0.2)

        # Reshape the features and labels for model input
        train_features = train_features.unsqueeze(-1)  # Shape: (num_samples, sequence_length, 1)
        test_features = test_features.unsqueeze(-1)    # Shape: (num_samples, sequence_length, 1)
        train_labels = train_labels.unsqueeze(-1)      # Shape: (num_samples, 1)
        test_labels = test_labels.unsqueeze(-1)        # Shape: (num_samples, 1)

        print(f"File SS{i + 1} - Train Features Shape: {train_features.shape}, Train Labels Shape: {train_labels.shape}")
        print(f"File SS{i + 1} - Test Features Shape: {test_features.shape}, Test Labels Shape: {test_labels.shape}\n")

        # Create DataLoader for batch processing
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for model_name in models_to_test:
            print(f"Training and evaluating {model_name} on file SS{i + 1}...")

            predictions, targets, sigmas, model = train_and_evaluate(model_name, model_category, train_loader, test_loader, device, epochs=epochs)

            # Save model
            model_save_path = os.path.join("models", f"{model_name}_SS{i + 1}.pth")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved model to {model_save_path}")

            # Save plot with confidence intervals
            plot_save_path = os.path.join("plots", f"{model_name}_Results_SS{i + 1}.png")
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
            plot_results(predictions, targets, sigmas, title=f"{model_name} Results on file SS{i + 1}", save_path=plot_save_path)
            print(f"Saved plot to {plot_save_path}")
