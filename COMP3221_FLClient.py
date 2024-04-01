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
import pandas


import matplotlib
import matplotlib.pyplot as plt

	
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

	plt.scatter(x_tensor[:, 0], y_tensor)
	plt.show()
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