import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import LinearRegressionModel

import socket
import sys
import json
import copy
import random
import pandas
import threading


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
		total_loss = 0
		epochs = 20
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
			total_loss += epoch_loss
			# print(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

		return total_loss / epochs

	def test(self, x_tensor, y_tensor):
		# Run the model against the test data
		# This with is included to stop the model from calculating the gradients
		# Saving time efficiency
		with torch.no_grad():
			mse = nn.MSELoss()
			output = self.model(x_tensor)
			loss = mse(output, y_tensor)
			return loss.item()


def send_parameters(client_id, model: LinearRegressionModel):
	sending_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_address = ('localhost', 6000)
	sending_socket.connect(server_address)

	sending_value = []
	for key, value in model.state_dict().items():
		sending_value.append((key, value.tolist()))
	data = {
		'client_id': client_id,
		'message_type': 'model',
		'data':	sending_value
	}
	sending_socket.send(json.dumps(data).encode())
	sending_socket.close()

def receive_parameters(model: LinearRegressionModel, client_id):
	receiving_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	receiving_socket.bind(('localhost', client_id))
	receiving_socket.listen(5)
	server_socket, server_address = receiving_socket.accept()
	data = server_socket.recv(1024).decode()
	data = json.loads(data)
	state_dict = {}
	if data['message_type'] == 'model':
		params = data['data']
		for key, value in params:
			state_dict[key] = torch.tensor(value)
		model.load_state_dict(state_dict)
	server_socket.close()
	receiving_socket.close()



	
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
	user = UserAVG(client_id, LinearRegressionModel(), 0.05, x_tensor.size(0))


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
		print(f"I am client {client_id[-1]}")
		# Receive the data
		receive_parameters(user.model, port)
		print(f"Received new global model")
		testing_loss = user.test(x_tensor, y_tensor)
		print(f"Testing MSE: {testing_loss}")

		# Train the model and update it
		training_loss = user.train(x_tensor, y_tensor)
		print(f"Training MSE: {training_loss}")
		print(f"Sending new local model")
		# Send the data
		send_parameters(client_id, user.model)


	# # Test the model
	# x_test_tensor, y_test_tensor = parse_csv_file("FLData/calhousing_test_client1.csv")

	# user.test(x_test_tensor, y_test_tensor)
	# Connect to the Server
		

	

if __name__ == "__main__":
	main()