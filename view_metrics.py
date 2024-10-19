import numpy as np
import matplotlib.pyplot as plt
import os
import timm
import torch

def view_confusion_matrix(epoch, base_path="Model_Training/confusion_matrix"):
    """
    Load and display the confusion matrix for a specific epoch.
    
    Args:
    epoch (int): Epoch number of the confusion matrix to load.
    base_path (str): Directory where confusion matrices are stored.
    """
    matrix_path = os.path.join(base_path, f"confusion_matrix_epoch_{epoch}.npy")
    if not os.path.exists(matrix_path):
        print(f"Confusion matrix for epoch {epoch} not found at {matrix_path}.")
        return
    
    cm = np.load(matrix_path)
    plt.figure(figsize=(10, 8))
    plt.matshow(cm, cmap='Blues', fignum=1)
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.show()
    input("Press Enter to continue...")
    plt.close()

def see_checkpoint_metrics(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    if not isinstance(checkpoint, dict):
        print(f"Invalid checkpoint format at {checkpoint_path}.")
        return
    
    # Exclude model and optimizer states
    excluded_keys = ['model_state_dict', 'optimizer_state_dict']
    short_style_checkpoint = {k: v for k, v in checkpoint.items() if k not in excluded_keys}
    
    print("Checkpoint Metrics:")
    for key, value in short_style_checkpoint.items():
        print(f"{key}: {value}")

    # wait for user input
    input("Press Enter to continue...")

see_checkpoint_metrics("Facial_Detection/best_models/59_best_epoch_acc_81.47_original_data_only_evaluated.pt")


# view_confusion_matrix(0)