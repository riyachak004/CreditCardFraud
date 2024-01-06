from tkinter import N
import numpy as np
from csv import reader
import array as arr
from numpy import sqrt
import math
import random

fraud = []
datalines = []
weights = []
hidden_weights = []
error = []
total_size = 0

#assign output values 
# output = np.array([0,1,1,0])
# output = output.reshape(4,1)
#print(output.shape)

#assign weights
# weights = np.array([[0.1],[0.2]])

# guassian function to assign intial weights - better than uniform becuase higher probability that number chosen is closer to
# mean which is 0 - and the lower the number the better. there is a possbility of getting a high number but decided to take the risk

# assign intial weights
# def assign_weights():
#     double = 29 * total_size
#     for i in range(double):
#         variance = 2.0/4 #input nodes + output node
#         sd = sqrt(variance)
#         weight = random.gauss(0,sd)
#         weights.append(weight)
#         if (len(hidden_weights) < total_size):
#             hidden_weights.append(weight)
        

# read file input
# def read_input():
#     with open("creditcard.csv", "r") as lines: 
#         csv_reader = reader(lines)
#         for row in csv_reader:
#             entry = ([row[1], row[2],row[3], row[4], row[5],row[6], row[7], row[8],row[9], row[10], row[11],row[12], row[13], row[14],row[15], row[16], row[17],row[18], row[19], row[20],row[21], row[22], row[23],row[24], row[25], row[26],row[27], row[28], row[29]])
#             datalines.append(entry)
#             fraud.append(row[30])


# sigmoid activation function 
# both sigmoid and tanh keep bounds to avoid the exploding graident problem 
#sigmoid used for a smaller learning rate - takes longer but most likely better
# with larger learning rates we are able to jump out of local minima to optimal minima
def sigmoid_func(x):
    if (x >= 7):
        return 1
    if (x <= -7):
        return 0
    return 1/(1+ math.exp(-x))

# derivative of sigmoid function - used in gradient descent 
def sig_der(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))


#ReLU activation function
def relu_func(x): 
    return max(x, 0)

# relu derivative 
def relu_der(x): 
    if (x < 0):
        return 0
    if (x >= 0):
        return 1


# update weights
def update_weights():
    weights_ar = np.array(weights, dtype=float)
    hidden_weights_ar = np.array(hidden_weights, dtype=float)
    bias_list = [0.3] * total_size
    bias_first_layer = np.array(bias_list, dtype=float)
    bias_second_layer = 0.3
    partial_derv = []
    weights_ar = weights_ar.reshape(29,total_size)
    hidden_weights_ar = hidden_weights_ar.reshape(total_size,1)
    fraud_arr = np.array(fraud, dtype=float)
    tracker = 0
    for epoches in range(1):
        tracker+=1
        for i in range(total_size): 
            hidden_nodes = []
            input_arr = np.array(datalines[i], dtype=float)
            input_arr = input_arr.reshape(1,29)
            weighted_sum = np.dot(input_arr, weights_ar) + bias_first_layer #first instance
            
            #calculating z for the hidden nodes
            for x in weighted_sum:
                for y in x:
                    y = relu_func(y)
                    hidden_nodes.append(y)
                
            #calculating z for output
            z = np.dot(weighted_sum, hidden_weights_ar) + bias_second_layer
            second_weighted_sum = sigmoid_func(z)
            error = second_weighted_sum - fraud_arr[i]

            # start with hidden row
            cost_value = 2 * error 
            value_total = sig_der(z)
            total_range = 28
            bias_second_layer = bias_second_layer - 0.07 * cost_value * value_total
            for x in range(len(hidden_weights_ar)):
                value = cost_value * value_total * hidden_weights_ar[x]
                partial_derv.append(value)
                hidden_weights_ar[x] = hidden_weights_ar[x]- 0.07 * hidden_nodes[x] * cost_value * value_total
                
            bias_first_layer_ar = np.array(bias_first_layer, dtype=float)
            partial_derv_ar = np.array(partial_derv, dtype=float)
            #print(i)
            for x in range(total_range):

                bias_first_layer_ar[x] = bias_first_layer_ar[x] - 0.07 * relu_der(hidden_nodes[x]) * partial_derv_ar[x]
                for y in range(total_range):
                    weights_ar[x][y] = weights_ar[x][y] - 0.07 * relu_der(hidden_nodes[x]) * partial_derv_ar[x] * input_arr[0][y]
    #print(weights_ar)
    test_weights(weights_ar, bias_first_layer, bias_second_layer, hidden_weights_ar)


def test_weights(weights_ar, bias_first_layer, bias_second_layer, hidden_weights_ar):
    new_fraud = []
    new_datalines = []
    hidden_nodes = []
    with open("copydata.csv", "r") as lines: 
        csv_reader = reader(lines)
        for row in csv_reader:
            entry = ([row[1], row[2],row[3], row[4], row[5],row[6], row[7], row[8],row[9], row[10], row[11],row[12], row[13], row[14],row[15], row[16], row[17],row[18], row[19], row[20],row[21], row[22], row[23],row[24], row[25], row[26],row[27], row[28], row[29]])
            new_datalines.append(entry)
            new_fraud.append(row[30])
    
    new_fraud_ar = np.array(new_fraud, dtype=int)
    new_total_size = len(new_fraud)
    print("SIZE" , new_total_size)
    correct = 0
    total_fraud = 0
    total_fraud_correct = 0
    for i in range(new_total_size): 
        input_arr = np.array(new_datalines[i], dtype=float)
        input_arr = input_arr.reshape(1,29)
        weighted_sum = np.dot(input_arr, weights_ar) + bias_first_layer

    #calculating z for the hidden nodes
        for x in weighted_sum:
            for y in x:
                y = relu_func(y)
                hidden_nodes.append(y)
            
        #calculating z for output
        z = np.dot(weighted_sum, hidden_weights_ar) + bias_second_layer
        second_weighted_sum = sigmoid_func(z)
        # print("VAL: ", second_weighted_sum)
        # print("SECOND VAL: ", new_fraud_ar[i])
        
        if(second_weighted_sum > 0.5):
            second_weighted_sum = 1
        else:
            second_weighted_sum = 0
        # print(second_weighted_sum)
        if (second_weighted_sum == new_fraud_ar[i]):
            # print("CORRECT")
            correct+= 1
        if (new_fraud_ar[i] == 1):
            total_fraud += 1
        if(second_weighted_sum == new_fraud_ar[i] & new_fraud_ar[i] == 1):
            # print("ALSO CORRECT")
            total_fraud_correct += 1
        
    print("TOTAL CORRECT: ", correct, "/", new_total_size)
    print("TOTAL FRAUD CORRECT: ", total_fraud_correct, "/", total_fraud)


# read_input()
# total_size = len(fraud)
assign_weights()

update_weights()



# pred = np.array([0,1])
# result = np.dot(pred, weights) + bias
# res = sigmoid_func(result)
# print(res)

