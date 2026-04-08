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

def train_model(train_dataset, val_dataset, epochs=10, batch_size=32, lr=0.005):
    device = get_hardware_device()
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(train_dataset.chunk_df['label'].unique())
    model = AudioClassifier(num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Switched to SGD to prevent DirectML 'lerp' warnings and stabilize learning
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, foreach=False)
    
    print(f"Starting training for {epochs} epochs on {device}...")
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # --- NORMALIZATION FIX ---
            # Squash exploding audio volumes to mean=0, std=1 to prevent NaN loss
            b_mean = batch_features.mean()
            b_std = torch.sqrt(((batch_features - b_mean) ** 2).mean())
            batch_features = (batch_features - b_mean) / (b_std + 1e-6)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                # Apply the same normalization to the validation data
                b_mean = batch_features.mean()
                b_std = torch.sqrt(((batch_features - b_mean) ** 2).mean())
                batch_features = (batch_features - b_mean) / (b_std + 1e-6)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")
        
    return model