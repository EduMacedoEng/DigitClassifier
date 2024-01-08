import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

class DigitClassifier(nn.Module):
    # Initialize the neural network layers
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Fully connected layer from input to first hidden layer
        self.fc2 = nn.Linear(128, 64)   # Fully connected layer from first to second hidden layer
        self.fc3 = nn.Linear(64, 10)    # Fully connected layer from second hidden layer to output

    # Define the forward pass
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation function to first hidden layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation function to second hidden layer
        x = F.log_softmax(self.fc3(x), dim=1)  # Apply log softmax to output layer for classification
        return x

    # Define the training loop
    def train_loop(self, trainloader, criterion, optimizer, epochs=5):
        self.train()  # Set the model to training mode
        for epoch in range(epochs):  # Loop through the dataset multiple times
            running_loss = 0
            for images, labels in trainloader:
                images = images.view(images.shape[0], -1)  # Flatten the images
                optimizer.zero_grad()  # Clear the gradients
                output = self(images)  # Compute model output
                loss = criterion(output, labels)  # Calculate loss
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights
                running_loss += loss.item()  # Aggregate loss
            print(f"Epoch {epoch+1}/{epochs}.. Training loss: {running_loss/len(trainloader)}")

    # Static method to load data
    @staticmethod
    def load_data(batch_size=64):
        # Define transforms for the training data
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5,), (0.5,))  # Normalize the tensors
        ])
        # Load the MNIST dataset
        trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
        # Return the data loader for the dataset
        return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Make a prediction and visualize the result
    def predict_and_view(self, trainloader):
        images, labels = next(iter(trainloader))  # Get a batch of images
        img = images[0].view(1, 784)  # Take the first image and flatten it
        with torch.no_grad():  # Turn off gradients for prediction
            logits = self.forward(img)  # Get the raw model output
        probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities

        # Visualization of the image and probabilities
        img = img.view(1, 28, 28)  # Reshape image for display
        img = transforms.ToPILImage()(img)  # Convert to PIL image
        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)  # Set up subplot
        ax1.imshow(img, cmap='gray')  # Display the image
        ax1.axis('off')  # Turn off axis
        ax2.barh(np.arange(10), probabilities[0])  # Plot the probabilities
        ax2.set_aspect(0.1)  # Set the aspect ratio
        ax2.set_yticks(np.arange(10))  # Set y-ticks to class numbers
        ax2.set_yticklabels(np.arange(10))  # Label y-ticks with class numbers
        ax2.set_title('Class Probability')  # Set title
        ax2.set_xlim(0, 1.1)  # Set x-axis limits
        plt.tight_layout()  # Layout plot nicely
