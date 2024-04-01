import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import LinearRegressionModel

import os
import socket
import sys
import json
import copy
import random
import pandas


import matplotlib
import matplotlib.pyplot as plt

class UserAVG():
	def __init__(self, client_id, model, learning_rate, batch_size):
		self.client_id = client_id
		self.model = model
		self.learning_rate = learning_rate
		self.batch_size = batch_size

	def train(self, x_tensor, y_tensor):
		# Create a DataLoader object
		dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
		train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

		# Define the loss function
		criterion = nn.MSELoss()

		# Define the optimizer
		optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

		# Train the model
		epochs = 100
		for epoch in range(epochs):
			epoch_loss = 0
			for x_batch, y_batch in train_loader:
				optimizer.zero_grad()
				output = self.model(x_batch)
				loss = criterion(output, y_batch)
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
			epoch_loss /= len(train_loader)
			print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

		return self.model

	
def parse_csv_file():
	df = pandas.read_csv('FLData/calhousing_train_client1.csv')

	# Extract the input features and target values
	x = df.drop('MedHouseVal', axis=1).values
	y = df['MedHouseVal'].values

	x_tensor = torch.tensor(x, dtype=torch.float32)
	y_tensor = torch.tensor(y, dtype=torch.float32)

	y_tensor = y_tensor.reshape(-1, 1)

	# Normalize the input values
	x_tensor = min_max_scaling(x_tensor)
	return x_tensor, y_tensor

def min_max_scaling(tensor):
    min_vals = tensor.min(dim=0, keepdim=True).values
    max_vals = tensor.max(dim=0, keepdim=True).values
    return (tensor - min_vals) / (max_vals - min_vals)

def main():
	
	# CLI info parsing
	if len(sys.argv) != 4:
		print("Usage: python3 COMP3221_FLClient.py <Client-id> <Port-Client> <Opt-Method>")
		return

	client_id = sys.argv[1]
	port = int(sys.argv[2])
	if port <= 6000 or port > 6005:
		print("Port number should be between 6001 and 6005")
		return

	opt_method = int(sys.argv[3])
	if opt_method < 0 or opt_method >= 2:
		print("Optimization method should be 0 or 1")
		return
	
	x_tensor, y_tensor = parse_csv_file()

	# Create a Linear Regression Model
	model = LinearRegressionModel(input_size=8)
	user = UserAVG(client_id, model, 0.01, 2808)
	user.train(x_tensor, y_tensor)
	return

	# Connect to the Server
	while True:
		try:
			client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			server_address = ('localhost', 6000)
			client_socket.connect(server_address)
			break
		except Exception as e:
			print(f"Error: {e}")
			continue

	print(f"Connected to the server on {server_address[0]}:{server_address[1]}")
		

	

if __name__ == "__main__":
	main()