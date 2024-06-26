import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import LinearRegressionModel

import os
import socket
import sys
import json
import numpy as np
import time
import threading
import signal
import random

import matplotlib
import matplotlib.pyplot as plt

NUM_CLIENTS = 5
MAX_TRAINING_ROUNDS = 40
Terminate = threading.Event()


#Server should do these:
#Send_parameters: Broadcast the global model to all clients
# Aggregate_parameters: aggregate new global model from local models of all clients
# Evaluate: evaluate the global model across all clients

def signal_handler(sig, frame):
	print("Ctrl+C pressed")
	Terminate.set()
	sys.exit(0)

class ReceivingThread(threading.Thread):
	def __init__(self, model_data, client_to_be_added):
		threading.Thread.__init__(self)
		self._stop = Terminate
		self.model_data = model_data
		self.client_to_be_added = client_to_be_added
	
	def stopped(self):
		return self._stop.is_set()

	def run(self):
		receiving_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		server_address = ('localhost', 6000)
		receiving_socket.bind(server_address)
		receiving_socket.listen(5)
		receiving_socket.settimeout(1)
		while True:
			if self.stopped():
				receiving_socket.close()
				return

			try:
				client_socket, client_address = receiving_socket.accept()
			except socket.timeout:
				continue

			data = client_socket.recv(1024).decode()
			data = json.loads(data)
			if data['message_type'] == 'init':
				self.client_to_be_added.append(data['data'])
				client_socket.close()
			elif data['message_type'] == 'model':
				params = data['data']
				state_dict = {}
				for key, value in params:
					state_dict[key] = torch.tensor(value)
				state_dict['client_id'] = data['client_id']
				self.model_data.append(state_dict)
				print(f"Getting local model from client {state_dict['client_id'][-1]}")
				client_socket.close()
			else:
				print("No known message type")
				client_socket.close()



def send_parameters(model: LinearRegressionModel, client_info):
	time.sleep(1)
	for client in client_info:
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			s.connect(('localhost', client['port']))
			sending_value = []
			for key, value in model.state_dict().items():
				sending_value.append((key, value.tolist()))
				data = {
					'message_type': 'model',
					'data':	sending_value
				}
			s.sendall(json.dumps(data).encode())
		except Exception as e:
			print(f"Error: {e}, {str(client)}")
			time.sleep(1)
			continue
		s.close()

# Receive the updated models from all clients and aggregate the updated models.

def aggregate_parameters(server_model, client_model_params, client_info, sub_client):
	for param in server_model.parameters():
		param.data = torch.zeros_like(param.data)
	
	data_sum = sum([c['data_size'] for c in client_info])
	if sub_client == 0:
		sub_client = NUM_CLIENTS

	# check for option 0 which means all clients
	if sub_client == 0:
		sub_client = len(client_info)

	for i in range(sub_client):
		if len(client_model_params) == 0:
			break
		client_model_data = random.choice(client_model_params)
		for client in client_info:
			if client['client_id'] == client_model_data['client_id']:
				for key in server_model.state_dict().keys():
					server_model.state_dict()[key].data += client_model_data[key].data.clone() * (client['data_size']/data_sum)
				for i, item in enumerate(client_model_params):
					if item['client_id'] == client_model_data['client_id']:
						client_model_params.pop(i)
						break
				break

	client_model_params.clear()
	return server_model


def initial_client_connections(port, client_info):

	# Create a socket object
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	# Bind the socket to a specific address and port
	server_address = ('localhost', port)
	server_socket.bind(server_address)

	# Listen for incoming connections
	server_socket.listen(NUM_CLIENTS)

	print(f"Server is listening on {server_address[0]}:{server_address[1]}")

	client_socket, client_address = server_socket.accept()
	print(f"Accepted connection from {client_address[0]}:{client_address[1]}")
	data = client_socket.recv(1024).decode()
	data = json.loads(data)
	if data['message_type'] == 'init':
		client_info.append(data['data'])
	else:
		print("Non INIT message received")

	timeout = 30

	# Set a timeout for the server socket
	server_socket.settimeout(1)
	start_time = time.time()
	while True:
		try:
			client_socket, client_address = server_socket.accept()
			print(f"Accepted connection from {client_address[0]}:{client_address[1]}")
			data = client_socket.recv(1024).decode()
			data = json.loads(data)
			if data['message_type'] == 'init':
				client_info.append(data['data'])
			else:
				print("Non INIT message received")
				continue
			client_socket.close()
		except socket.timeout:
			current_time = time.time()
			if current_time - start_time >= timeout:
				break
		except Exception as e:
			print(f"Error: {e}")
			client_socket.close()
			break

	
	# Close the server socket
	server_socket.close()


def main():
	signal.signal(signal.SIGINT, signal_handler)

	# CLI info parsing
	if len(sys.argv) != 3:
		print("Usage: python3 COMP3221_FLServer.py <port-Server> <Sub-Client>")
		return
	
	port = int(sys.argv[1])
	if port != 6000:
		print("Port number should be 6000")
		return
	
	sub_client = int(sys.argv[2])
	if sub_client >= NUM_CLIENTS:
		print("Sub-client number should be less than 5")
		return

	# Server should generate a global linear regression model HERE


	# Listen for initial connection requests from clients
	client_info = []     #{'client_id': 'client2', 'port': 6002, 'data_size': 2476}
	initial_client_connections(port, client_info)

	global_model = LinearRegressionModel()
	# Send the global model to all clients

	training_round = 1
	client_model_params = []
	clients_to_be_added = []
	ReceivingThread(client_model_params, clients_to_be_added).start()
	send_parameters(global_model, client_info)
	while training_round <= MAX_TRAINING_ROUNDS:

		print(f"Global Iteration {training_round}:")
		print(f"Total Number of clients {len(client_info)}")

		# Add any new clients waiting to be added to the system

		while len(client_model_params) < len(client_info):
			pass

		# Aggregate the updated models from all the clients
		print("Aggregating new global model")
		aggregate_parameters(global_model, client_model_params, client_info, sub_client)

		for client in clients_to_be_added:
			client_info.append(client)
		clients_to_be_added.clear()

		print("Broadcasting new global model")
		send_parameters(global_model, client_info)
		training_round += 1

	Terminate.set()





if __name__ == "__main__":
	main()