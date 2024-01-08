# %%
import torch
import torch.nn as nn
from digit_classifier import DigitClassifier

# initialize the network
model = DigitClassifier()

# Define the loss
criterion = nn.CrossEntropyLoss()

# Define the loss function as negative log likelihood loss
criterion_negative = nn.NLLLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

# Load data
trainloader = model.load_data()

# Train the model
model.train_loop(trainloader, criterion_negative, optimizer, epochs=10)
model.predict_and_view(trainloader)
# %%
