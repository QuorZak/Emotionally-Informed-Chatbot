import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import timm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from dataset import CustomImageDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea

class Trainer:
    def __init__(self):
        """
        Initialize the Trainer with data transformations, datasets, model, and optimization components.
        """
        
        ############  Control all of training paramters here  ############################
        self.image_source = "ed_lab" # If you want to use a specific dataset, change this to the dataset name
        self.run_name = "us_plus_lab_vgg16" # Used to differentiate between different training runs
        self.lr = 0.0001  # Learning rate. First pass was 0.01.
        self.batch_size = 10
        self.use_scheduler = False  # Whether to use a learning rate scheduler
        self.num_epochs = 200

        # Doesn't do any training and gets Metrics for a single checkpoint
        self.evaluate_only = True
        self.evaluate_only_model_path = "Facial_Detection/best_models/59_best_epoch_acc_81.47_original_data_only.pt"
        self.evaluate_only_epoch = 59

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
            transforms.ToTensor(),  # Convert images to PyTorch tensors (scales values to [0, 1])
            transforms.Normalize((0.5,), (0.5,))  # Normalize with mean=0.5 and std=0.5 (scales to [-1, 1])
        ])

        # Define the data locations
        self.train_data_location = f'./inputs_and_outputs/{self.image_source}_train_images'
        self.test_data_location = f'./inputs_and_outputs/{self.image_source}_test_images'

        # Initialise the model
        #self.model = timm.create_model('timm/convnextv2_pico.fcmae_ft_in1k', pretrained=False)
        self.model = models.vgg16(pretrained=False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        label_list_dir =  os.path.join(current_dir, self.train_data_location)
        num_classes = len(os.listdir(label_list_dir))
        #self.model.head.fc = nn.Linear(512, num_classes) # CONVNEXT
        self.model.classifier[6] = nn.Linear(4096, num_classes) # VGG16

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Learning rate scheduler for dynamic learning rate adjustment
        if (self.use_scheduler):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=7)

        ####################   End of training parameters  ###############################

        # Define the locations for saving checkpoints and confusion matrices
        self.output_set = f'{self.run_name}_In-{self.image_source}'  # Input + output = final name
        self.short_save_path = f'Model_Training/inputs_and_outputs/{self.output_set}_short_checkpoints'
        self.full_save_path = f'Model_Training/inputs_and_outputs/{self.output_set}_checkpoints'
        self.best_checkpoints_location = f'Facial_Detection/{self.output_set}_checkpoints'
        self.best_cm_save_path = f'Facial_Detection/{self.output_set}_confusion_matrix'

        # Create directories for saving checkpoints and confusion matrices
        os.makedirs(self.short_save_path, exist_ok=True)
        os.makedirs(self.full_save_path, exist_ok=True)
        os.makedirs(self.best_checkpoints_location, exist_ok=True)
        os.makedirs(self.best_cm_save_path, exist_ok=True)

        # Create datasets and data loaders
        self.train_dataset = CustomImageDataset(root_dir=self.train_data_location, transform=self.transform)
        self.test_dataset = CustomImageDataset(root_dir=self.test_data_location, transform=self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)  # Move model to the selected device
        
        # Define loss function
        # CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one single class
        # Weights are inversely proportional to the class frequencies
        class_counts = self.train_dataset.classes_count
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        # Initialize epoch to 0
        self.start_epoch = 0

        # Some variables to help manage the training process
        self.previous_accuracy = 0.0
        self.previous_loss = 0.0
        self.best_epoch_and_accuracy = (0, 0.0)

    def save_checkpoint(self, epoch, accuracy, f1, precision, recall, is_best=False):
        """
        Save the model checkpoint in both the short and full formats, including F1 score, precision, and recall.
        
        Args:
        epoch (int): Current epoch number.
        accuracy (float): Accuracy of the model.
        f1 (float): F1 score.
        precision (float): Precision score.
        recall (float): Recall score.
        is_best (bool): Whether this checkpoint is the best-performing model.
        """
        if not is_best:
            # Save short-style checkpoint (state_dict)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
            if self.evaluate_only: #replace .pt with _evaluated.pt
                torch.save(checkpoint, self.evaluate_only_model_path.replace(".pt", "_evaluated.pt"))
            torch.save(checkpoint, os.path.join(self.short_save_path, f"{epoch}.pt"))
        
            # Save full model
            torch.save(self.model, os.path.join(self.full_save_path, f"{epoch}.pt"))
        else:
            torch.save(self.model, os.path.join(self.best_checkpoints_location, f"{epoch}.pt"))

    
    def load_checkpoint_if_exists(self):
        """
        Load the latest checkpoint if it exists in the save_path directory.
        Tries to load short-format checkpoints first, then full-format if necessary.
        """
        if self.evaluate_only:
            self.model = torch.load(self.evaluate_only_model_path)
            self.start_epoch = self.evaluate_only_epoch
            print(f"Loaded model for evaluation only: {self.evaluate_only_model_path}, Epoch: {self.start_epoch}")
            return

        checkpoints = [f for f in os.listdir(self.short_save_path) if f.endswith(".pt")]
        full_checkpoints = [f for f in os.listdir(self.full_save_path) if f.endswith(".pt")]
        
        # Try loading short-format checkpoint first
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))  # Get highest epoch number
            checkpoint_path = os.path.join(self.short_save_path, latest_checkpoint)
            
            print(f"Loading short-format checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)  # Only works for TIMM models

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
            
                print(f"Previous short-epoch checkpoint found. Starting training on epoch {self.start_epoch}")
        
        # If no short-format checkpoints, try full model checkpoints
        elif full_checkpoints:
            latest_full_checkpoint = max(full_checkpoints, key=lambda x: int(x.split('.')[0]))  # Get highest epoch number
            checkpoint_path = os.path.join(self.full_save_path, latest_full_checkpoint)
            
            print(f"Loading full model checkpoint: {checkpoint_path}")
            self.model = torch.load(checkpoint_path)
            
            # Extract epoch from the filename (full checkpoint)
            self.start_epoch = int(latest_full_checkpoint.split('.')[0]) + 1  # Resume from the next epoch
            
            print(f"Previous epoch checkpoint found. Starting training on epoch {self.start_epoch}")
        
        else:
            print("No checkpoint found. Starting training from scratch.")

        print("\n")

    def save_confusion_matrix(self, epoch, labels, predictions, is_best=False):
        """
        Save the confusion matrix for the given epoch.
        
        Args:
        epoch (int): The current epoch number.
        labels (list): Ground truth labels.
        predictions (list): Model predictions.
        """
        cm = confusion_matrix(labels, predictions)
        
        # Get class names from the dataset
        class_names = self.train_dataset.classes

        # Plot confusion matrix with seaborn
        plt.figure(figsize=(10, 8))
        sea.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix for Epoch {epoch}')
    
        # Change the save path based on whether this is the best model
        save_path = self.best_cm_save_path

        if self.evaluate_only:
            save_path = self.evaluate_only_model_path.replace(".pt", "")
            plt.savefig(save_path + "_cm.png")
            plt.close()
            return

        # Save the confusion matrix plot
        plt.savefig(os.path.join(save_path, f"{epoch}_cm.png"))
        plt.close()  # Close the figure to prevent it from displaying

    def train(self):
        """
        Train the model for a specified number of epochs, resuming from the latest checkpoint if available.
        
        Args:
        num_epochs (int): Number of epochs to train for.
        """
        self.load_checkpoint_if_exists()  # Load the latest checkpoint if exists

        if self.evaluate_only:
            self.start_epoch, self.num_epochs = self.start_epoch,self.start_epoch+1 # do it once

        for epoch in range(self.start_epoch, self.num_epochs):
            if not self.evaluate_only:
                self.model.train()  # Set model to training mode (enables dropout, batch norm updates, etc.)
            running_loss = 0.0
            average_loss = 0.0
            
            if not self.evaluate_only:
                # Training loop
                for images, labels in tqdm(self.train_loader):
                    # Move data to the selected device
                    images, labels = images.to(self.device), labels.to(self.device)
                
                    self.optimizer.zero_grad()  # Zero the parameter gradients
                
                    outputs = self.model(images)  # Forward pass: compute predicted outputs by passing inputs to the model
                
                    # Calculate the loss
                    loss = self.criterion(outputs, labels)
                
                    loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
                    self.optimizer.step()  # Perform a single optimization step (parameter update)
                
                    running_loss += loss.item()  # Accumulate the loss
            
                # Print average loss for the epoch
                average_loss = running_loss / len(self.train_loader)
                print("\n" + f"Epoch {epoch}, Loss: {average_loss}")
            
            # Evaluation loop
            self.model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm, etc.)
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            with torch.no_grad():  # Disable gradient computation for efficiency during evaluation
                for images, labels in tqdm(self.test_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    predicted = torch.argmax(outputs.data, 1)  # Get the index of the max log-probability
                    predicted_labels = torch.argmax(labels.data, 1)  # Convert one-hot to class indices
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    total += labels.size(0)
                    correct += (predicted == predicted_labels).sum().item()

            # Convert all_labels to a Tensor
            all_labels = torch.tensor(np.array(all_labels))
            all_labels = torch.argmax(all_labels, axis=1) # Convert one-hot encoded labels to class indices
            accuracy = (correct / total) * 100
            precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            print(f"Accuracy on test set: {accuracy}% ")
            # print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")

            if self.evaluate_only:
            # Calculate and save the confusion matrix.
                self.save_confusion_matrix(epoch, all_labels.cpu().numpy(), all_predictions)
            
            # Save model checkpoint
            self.save_checkpoint(epoch, accuracy, f1, precision, recall)

            # Check if the model is the new best model
            if accuracy > self.best_epoch_and_accuracy[1]:
                self.best_epoch_and_accuracy = (epoch, accuracy)
                self.save_checkpoint(f"{epoch}_best_acc_{accuracy:.2f}_avg_loss_{average_loss:.4f}", accuracy, f1, precision, recall, is_best=True)
                self.save_confusion_matrix(epoch, all_labels.cpu().numpy(), all_predictions, is_best=True)

            # Define early stopping criteria
     #       if abs(accuracy-self.previous_accuracy)<0.0001 and abs(average_loss-self.previous_loss)<0.000001:
      #          print("Accuracy and loss change less than 0.0001% and 0.000001. Stopping training.")
     #           break

            self.previous_accuracy = accuracy
            self.previous_loss = average_loss

            # Update the learning rate based on the validation loss.
            if (self.use_scheduler):
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(average_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"Learning rate updated to {new_lr}")

# Usage
trainer = Trainer()
trainer.train()