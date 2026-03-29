# gizzbrain/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_hardware_device():
    """Dynamically routes tensors to the correct hardware (Mac M-Series, PC CUDA/AMD, or CPU)."""
    if torch.cuda.is_available():
        print("Hardware allocated: CUDA/ROCm")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Hardware allocated: Apple MPS")
        return torch.device("mps")
    else:
        try:
            import torch_directml
            if torch_directml.is_available():
                print("Hardware allocated: AMD DirectML")
                return torch_directml.device()
        except ImportError:
            pass
            
    print("Hardware allocated: CPU")
    return torch.device("cpu")

class AudioClassifier(nn.Module):
    """
    Convolutional Neural Network for identifying audio spectrograms.
    """
    def __init__(self, num_classes, time_steps=216):
        super(AudioClassifier, self).__init__()
        
        # Conv layer: looks for local patterns in the spectrogram
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the flattened size dynamically based on input dimensions
        # MaxPool2d(2,2) halves the height (128 -> 64) and the width (time_steps -> time_steps // 2)
        linear_input_size = 32 * 64 * (time_steps // 2)
        
        # Flattening and Linear layers to output the final prediction
        self.fc1 = nn.Linear(linear_input_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
        return x

def train_model(dataset, epochs=10, batch_size=32, lr=0.001):
    """
    The main training loop. The CLI will call this function.
    """
    device = get_hardware_device()
    
    # 1. Setup DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Determine number of unique classes from the dataset
    num_classes = len(dataset.chunk_df['label'].unique())
    
    # 3. Initialize Model, Loss, and Optimizer
    model = AudioClassifier(num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training for {epochs} epochs on {device}...")
    
    # 4. The Loop (To be filled out next)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward, Backward, Optimize logic will go here
            pass
            
        print(f"Epoch {epoch+1}/{epochs} completed.")
        
    return model