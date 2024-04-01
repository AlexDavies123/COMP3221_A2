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
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt

NUM_CLIENTS = 5


#Server should do these:
#Send_parameters: Broadcast the global model to all clients
# Aggregate_parameters: aggregate new global model from local models of all clients
# Evaluate: evaluate the global model across all clients


def send_parameters():
	pass

def aggregate_parameters():
	pass

def evaluate():
	pass


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
	client_info.append((client_socket, client_address))

	timeout = 10

	# Set a timeout for the server socket
	server_socket.settimeout(1)
	start_time = time.time()
	while True:
		try:
			client_socket, client_address = server_socket.accept()
			print(f"Accepted connection from {client_address[0]}:{client_address[1]}")
			data = client_socket.recv(1024)
			client_info.append(data)
		except socket.timeout:
			current_time = time.time()
			if current_time - start_time >= timeout:
				break
		except Exception as e:
			print(f"Error: {e}")
			break

	# Close the server socket
	server_socket.close()


def main():

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
	client_info = []
	initial_client_connections(port, client_info)
	print(client_info)

	model = LinearRegressionModel()
	torch.deserailization



if __name__ == "__main__":
	main()