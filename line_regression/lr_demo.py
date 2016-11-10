#coding=utf8
'''
Title: Liner Regression with Gradient Descent Optimization
Version: V0.2
Author: Louie Wang
Date: Last update 2015-9-10
'''
import numpy as np
import matplotlib.pyplot as plt

def load_file(filename):
    data = np.loadtxt(filename, dtype='float64')
    ones = np.ones(data.shape[0])
    return np.c_[ones, data]

def load_Y_file(filename):
    data = np.array([float(line.strip().split()[0]) for line in open(filename, 'r').readlines()])
    return data

def hypthesis(thetas, X):
    return X.dot(thetas)

def sum_J(X, Y, thetas):
    sum_j = sum(np.power(hypthesis(thetas, X) - Y, 2))
    return (1.0/2) *sum_j

def get_gradient(X, Y, thetas):
    X, Y, thetas = np.array(X), np.array(Y), np.array(thetas)
    grads = X.T.dot(X.dot(thetas) - Y)
    return grads

def get_X_Y(X, Y, thetas):
    return [x[1]+2000 for x in X], [y*1000 for y in Y], [hypthesis(thetas, X[i])*1000 for i in range(len(X))]

def show_plot(sum_J, X, Y, thetas, iteration):
    x_paint, y_paint, hypthesis = get_X_Y(X, Y, thetas)
    axarr[0].cla()
    axarr[0].plot(x_paint, y_paint, '*', x_paint, hypthesis, 'r')
    axarr[0].set_xlim([2000, 2014])
    axarr[0].set_ylim([0,15000])
    axarr[0].set_title('Liner Regression with Gradient Descent Optimization')
    axarr[0].set_xlabel('Year')
    axarr[0].set_ylabel('Housing Average Price')
    axarr[1].set_title('The convergence curve of cost function')
    axarr[1].scatter(iteration, sum_J, color='r')
    axarr[1].set_xlabel('Iteration steps')
    axarr[1].set_ylabel('Cost value')
    axarr[1].set_xlim([1, 50])
    axarr[1].set_ylim([0, 500])
    fig.canvas.draw()

def gradient_decient(X, Y, thetas, rate, end_condition):
    sum_old, sum_new = 0, sum_J(X, Y, thetas)
    iteration = 0
    while iteration <500 and abs(sum_old -sum_new) > end_condition:
        show_plot(sum_new, X, Y, thetas, iteration)
        grads = get_gradient(X, Y, thetas)
        thetas -= rate * grads
        sum_old= sum_new
        sum_new = sum_J(X, Y, thetas)
        iteration+=1
        print("iteration%d:The hypothesis is y=%f + %f*x  Cost:%f" % (iteration, thetas[0], thetas[1], sum_new))
    return thetas

def predict(X, model):
    data, ones = np.array(X), np.ones(len(X))
    X = np.c_[ones, data]
    return hypthesis(model, X)

if __name__ =="__main__":
    fig, axarr = plt.subplots(1, 2, figsize=(14,5))
    fig.show()
    end_condition = 1e-5
    learning_rate = 0.0002
    init_thetas = [1, -0.5]
    X, Y =load_file(r'data/X.dat'), load_Y_file(r'data/Y.dat')
    model = gradient_decient(X, Y, init_thetas, learning_rate, end_condition)
    print predict([[14],[15]], model)
    raw_input()