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
import pickle


import matplotlib
import matplotlib.pyplot as plt

class UserAVG():
	def __init__(self, client_id, model, learning_rate, batch_size):
		# ID value of the client, will need it to print the client name
		self.client_id = client_id
		# The model that the client will be training
		self.model = model
		# The learning rate of the model (Can be changed to whatever we want)
		self.learning_rate = learning_rate
		# The batch size of the model (Determines what sort of optimisation we want to do)
		self.batch_size = batch_size

	def train(self, x_tensor, y_tensor):
		# Create the dataset from the two tensors
		dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
		# Create the data loader from the dataset, will load the amount of data specified by batch_size
		train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

		# Define the loss function, we r using MSELOSS (Determines how close to the desired result we are)
		mse = nn.MSELoss()

		# Define the optimizer, we r using SGD (Stochastic Gradient Descent) This can and im pretty sure should be changed
		# Basically is how the model will have it paramters updated
		optimiser = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

		# Train the model
		# 100 Iterations of the dataset
		epochs = 100
		for epoch in range(epochs):
			epoch_loss = 0
			# For the batch in the train loader
			for x_batch, y_batch in train_loader:
				# Needs to be done to clear the gradients of the model
				optimiser.zero_grad()
				# This is where the model is run against the data
				output = self.model(x_batch)
				# Calculates the loss of the model based on MSELOSS
				loss = mse(output, y_batch)
				# Calculates the gradients of the model
				loss.backward()
				# Updates the model parameters based on the gradients calculated above (stored in the tensors)
				optimiser.step()
				# Calculate the total loss of the epoch
				epoch_loss += loss.item()
			# Calculate the total loss of the epoch based on how many batches there was
			epoch_loss /= len(train_loader)
			print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

		return self.model

	def test(self, x_tensor, y_tensor):
		# Run the model against the test data
		# This with is included to stop the model from calculating the gradients
		# Saving time efficiency
		with torch.no_grad():
			mse = nn.MSELoss()
			output = self.model(x_tensor)
			loss = mse(output, y_tensor)
			print(f"Client {self.client_id} - Test Loss: {loss.item()}")


def send_parameters(model: LinearRegressionModel):
	sending_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_address = ('localhost', 6000)
	sending_socket.connect(server_address)
	sending_socket.send(pickle.dumps(model.state_dict()))
	sending_socket.close()

def receive_parameters(model: LinearRegressionModel, client_id):
	receiving_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_address = ('localhost', client_id)
	receiving_socket.bind(server_address)
	receiving_socket.listen(1)
	server_socket, server_address = receiving_socket.accept()
	data = server_socket.recv(1024)
	data = pickle.loads(data)
	model.load_state_dict(data)



	
def parse_csv_file(input_file):
	df = pandas.read_csv(input_file)

	# Extract the input features and target values
	x = df.drop('MedHouseVal', axis=1).values
	y = df['MedHouseVal'].values

	x_tensor = torch.tensor(x, dtype=torch.float32)
	y_tensor = torch.tensor(y, dtype=torch.float32)

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
	
	file_name = "FLData/calhousing_train_" + client_id + ".csv"
	
	x_tensor, y_tensor = parse_csv_file(file_name)
	user = UserAVG(client_id, LinearRegressionModel(), 0.01, x_tensor.size(0))


	# # Test the model
	# x_test_tensor, y_test_tensor = parse_csv_file("FLData/calhousing_test_client1.csv")

	# user.test(x_test_tensor, y_test_tensor)
	# Connect to the Server
	while True:
		try:
			sending_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			server_address = ('localhost', 6000)
			sending_sock.connect(server_address)
			break
		except Exception as e:
			print(f"Error: {e}")
			continue

	print(f"Connected to the server on {server_address[0]}:{server_address[1]}")
	data = {'message_type': 'init',
		   	 'data': {
				"client_id": client_id,
				"port": port,
				"data_size": x_tensor.size(0)
	}}
	sending_sock.sendall(json.dumps(data).encode())
	sending_sock.close()


	while True:
		# Receive the data
		receive_parameters(user.model, port)
	
		# Train the model and update it
		user.train(x_tensor, y_tensor)

		# Send the data
		send_parameters(user.model)


	# # Test the model
	# x_test_tensor, y_test_tensor = parse_csv_file("FLData/calhousing_test_client1.csv")

	# user.test(x_test_tensor, y_test_tensor)
	# Connect to the Server
		

	

if __name__ == "__main__":
	main()