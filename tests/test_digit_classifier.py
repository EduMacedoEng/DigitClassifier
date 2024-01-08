import unittest
import torch
from digit_classifier import DigitClassifier

class TestDigitClassifier(unittest.TestCase):
    
    def setUp(self):
        # Initialize the classifier and a set of known weights for reproducibility
        self.model = DigitClassifier()
        self.model.fc1.weight.data.fill_(0.01)
        self.model.fc2.weight.data.fill_(0.01)
        self.model.fc3.weight.data.fill_(0.01)
        self.model.fc1.bias.data.fill_(0)
        self.model.fc2.bias.data.fill_(0)
        self.model.fc3.bias.data.fill_(0)

    def test_forward(self):
        # Create a mock data point (28x28 image of all zeros)
        mock_img = torch.zeros((1, 28*28))
        
        # Forward pass through the network
        logits = self.model.forward(mock_img)
        
        # Assert the output is of the correct shape
        self.assertEqual(logits.shape, (1, 10))
        
        # Assert the output is a tensor
        self.assertIsInstance(logits, torch.Tensor)

    # Add more tests for other functionalities if needed

# This allows the test case to be run from the command line
if __name__ == '__main__':
    unittest.main()