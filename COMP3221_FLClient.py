import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import socket
import sys
import json
import copy
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size = 1):
        super(LinearRegressionModel, self).__init__()
        # Create a linear transformation to the incoming data
        self.linear = nn.Linear(input_size, 1)

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Apply linear transformation
        output = self.linear(x)
        return output.reshape(-1)
    

def main():
    # Generate random parameters for the linear regression model
    
    server = LinearRegressionModel(input_size=1)   

    # Listen for initial connection requests from clients
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', sys.argv[1]))
        s.listen(1)

        try:
            conn, addr = s.accept()
            print("Connection from: ", addr)
        except socket.timeout:
            continue

    

if __name__ == "__main__":
	main()