import torch
import matplotlib.pyplot as plt
import os

# Path where the checkpoints are saved
checkpoint_dir = 'C:/Users/Zak/source/repos/CS731-2024/emotional-aware-chatbot-cs731-g02/Model_Training/archive_short_checkpoints'

# Lists to store epochs and accuracies
epochs = []
accuracies = []

files = sorted(os.listdir(checkpoint_dir))

# Iterate through the saved checkpoint files
for i in range(16,62):
    # Load the checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_dir, str(i) +'.pt'))
        
    # Extract epoch and accuracy
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
        
    # Store them in the lists
    epochs.append(epoch)
    accuracies.append(accuracy)

# Plot accuracy per epoch
plt.plot(epochs, accuracies, label='Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.xlim(0, 70)  # Set x-axis limits
plt.ylim(50, 90)  # Set y-axis limits
plt.show()