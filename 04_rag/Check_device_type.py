import torch
if torch.backends.mps.is_available():
    print("MPS is available. Your Mac is using an Apple Silicon GPU.")
else:
    print("MPS is not available. Using CPU instead.")
        
# Check the device name being used
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")