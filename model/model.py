import torch.nn as nn
import numpy as np

# Define the DQN model architecture (must match the saved model)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

def preprocess_obs(obs):
    # Flatten the observation array
    return np.array(obs).flatten()