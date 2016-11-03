#coding=utf8
'''
Title: Liner Regression with Gradient Descent Optimization
Version: V0.1
Author: Louie Wang
Date: Last update 2016-9-10
'''
import math
import matplotlib.pyplot as plt

def read_X_file(filename):
    return [[1.0] + map(float, line.strip().split()) for line in open(filename, 'r').readlines()]

def read_Y_file(filename):
    return [float(line.strip().split()[0]) for line in open(filename, 'r').readlines()]

def hypthesis(thetas, x, y):
    h = 0
    for i in range(len(thetas)):
        h += thetas[i] * x[i]
    return h - y

def sum_J(X, Y, thetas):
    sum_j = 0
    for i in range(len(X)):
        sum_j += math.pow(hypthesis(thetas, X[i], Y[i]), 2)
    return (1.0/2) *sum_j

def get_gradient(X, Y, thetas):
    grad_0, grad_1 = 0, 0
    for i in range(len(X)):
        grad_0 += hypthesis(thetas, X[i], Y[i])
        grad_1 += hypthesis(thetas, X[i], Y[i]) * X[i][1]
    return [grad_0, grad_1]

def predict(X, model):
    return [model[0] + model[1] * x for x in X]

def gradient_decient(X, Y, thetas, rate, end_condition):
    sum_old, sum_new = 0, sum_J(X, Y, thetas)
    iterator = 0

    while iterator <500 and abs(sum_old -sum_new) > end_condition:
        grads = get_gradient(X, Y, thetas)
        for i in range(len(thetas)):
            thetas[i] -= rate * grads[i]
        sum_old= sum_new
        sum_new = sum_J(X, Y, thetas)
        iterator+=1
        
        print("iteration%d:The hypothesis is y=%f + %f * x  Cost:%f" % (iterator, thetas[0], thetas[1], sum_new))
    return thetas

if __name__ =="__main__":
    end_condition = 1e-5
    learning_rate = 1e-3
    init_thetas = [1, -0.5]
    X, Y =read_X_file(r'data/X.dat'), read_Y_file(r'data/Y.dat')

    model = gradient_decient(X, Y, init_thetas, learning_rate, end_condition)
    print predict([14], model)