import numpy as np
from csv import reader
from numpy import sqrt
import math

#Variables 
fraud = []
datalines = []

# Reads creditcard.csv and processes data
# Datalines contains data entries from index 1 to one before last
# Last element in each data entry is fraud (0/1) - stored in fraud list
def read_input():
    with open("training.csv", "r") as lines: 
        csv_reader = reader(lines)
        for row in csv_reader:
            fraud.append(int(row[-1]))
            datalines.append(row[1:-1])

# Assign intial weights with Glorot Xavier initialization
def assign_weights(input_no, output_no):
    variance = 1 / (input_no + output_no)
    sd = sqrt(variance)
    weights = np.random.normal(loc=0.0, scale=sd, size=(input_no, output_no))
    return weights

# sigmoid activation function 
def sigmoid_func(x):
    if x <= -100: 
        return 0
    return 1/(1+ math.exp(-x))

# Derivative of sigmoid function - used in gradient descent 
def sig_der(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))

# ReLU activation function
def relu_func(x): 
    return np.maximum(x, 0)

# ReLU derivative 
def relu_der(x): 
    if (x < 0):
        return 0
    if (x >= 0):
        return 1

# Conducts forward propagation
# Returns outcome and hidden_node weights
def forward_propagation(hidden_nodes, first_layer, input_arr, second_layer, bias_first_layer, bias_second_layer): 
    # Finding middle weights, dot product first
    middle_weights = np.dot(first_layer, input_arr) + bias_first_layer

    # Apply ReLU function to weights - has dim 29,1
    hidden_nodes = np.apply_along_axis(relu_func, 1, middle_weights)

    # Multiply hidden nodes with second_layer + bias
    z = np.dot(hidden_nodes.T, second_layer) + bias_second_layer
    return z[0][0], hidden_nodes

# update weights using stochastic gradient descent 
def neural_network(hidden_nodes, total_size, first_layer, second_layer, bias_first_layer, bias_second_layer):
    for x in range(5):
        for i in range(total_size):
            # print(i)
            # Forward pass 
            input_arr = np.array(datalines[i], dtype=float).reshape(29, 1)

            outcome, hidden_nodes = forward_propagation(hidden_nodes, first_layer, input_arr, second_layer, bias_first_layer, bias_second_layer)
            # Apply sigmoid func
            second_weighted_sum = sigmoid_func(outcome)

            # Backward pass
            error = second_weighted_sum - int(fraud[i])
            cost_value = 2 * error
            value_total = sig_der(outcome)

            learning_rate = 0.07
            # Update second layer
            for index, x in enumerate(second_layer): 
                second_layer[index][0] -= learning_rate * cost_value * value_total * hidden_nodes[index][0]
            bias_second_layer -= learning_rate * cost_value * value_total

            # Update first layer
            for x in range(29): 
                value = cost_value * value_total * second_layer[x][0] * relu_der(hidden_nodes[x])
                bias_first_layer[x] -= learning_rate * value
                input = input_arr * value 
                for y in range(29): 
                    first_layer[x][y] = first_layer[x][y] - input[y][0]
    test_weights(hidden_nodes, first_layer, second_layer, bias_first_layer, bias_second_layer)
                    

# Runs testing.csv to classify input based on neural network 
def test_weights(hidden_nodes, first_layer, second_layer, bias_first_layer, bias_second_layer):
    new_fraud = []
    new_datalines = []
    total_fraud = 0

    with open("testing.csv", "r") as lines:
        csv_reader = reader(lines)
        for row in csv_reader:
            new_datalines.append(row[1:-1])
            new_fraud.append(int(row[-1]))

    new_total_size = len(new_fraud)

    correct = 0
    total_fraud_correct = 0

    for i in range(new_total_size):
        input_arr = input_arr = np.array(new_datalines[i], dtype=float).reshape(29, 1)
        outcome, hidden_nodes = forward_propagation(hidden_nodes, first_layer, input_arr, second_layer, bias_first_layer, bias_second_layer)
        outcome = sigmoid_func(outcome)
        outcome = 0 if outcome < 0.5 else  1

        if new_fraud[i] == 1:
            total_fraud += 1
        if outcome == new_fraud[i]:
            correct += 1
        if outcome == new_fraud[i] == 1:
            total_fraud_correct += 1

    print("TOTAL CORRECT: ", correct, "/", new_total_size)
    print("TOTAL FRAUD CORRECT: ", total_fraud_correct, "/", total_fraud)
    

read_input()
total_size = len(fraud)

# Middle layer has 29 nodes
hidden_weights_first_layer = assign_weights(29,29)
hidden_weights_second_layer = assign_weights(29,1)
bias_first_layer = np.full((29, 1), 0.3)
bias_second_layer = 0.3
first_layer = np.array(hidden_weights_first_layer, dtype=float).reshape(29, 29)
second_layer = np.array(hidden_weights_second_layer, dtype=float).reshape(29, 1)
hidden_nodes = []

neural_network(hidden_nodes, total_size, first_layer, second_layer, bias_first_layer, bias_second_layer)
