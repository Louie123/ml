#coding=utf8
'''
Title: Logistic Regression with Gradient Descent Optimization
Version: V0.2
Author: Louie Wang
Date: Last update 2015-9-15
'''
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_X_file(filename):
    data = np.loadtxt(filename, dtype='float64')
    ones = np.ones(data.shape[0])
    return np.c_[ones, data]

def load_Y_file(filename):
    data = np.array([float(line.strip().split()[0]) for line in open(filename, 'r').readlines()])
    return data

def hypothesis(thetas, X):
    return 1.0/ (1 + np.exp(-1 * X.dot(thetas)))

def mle(X, Y, thetas):
    f = hypothesis(thetas, X)
    return Y.dot(np.log(f))+ (1 - Y).dot(np.log(1- f + 1e-10))

def get_gradient(X, Y, thetas ,i):
    error = hypothesis(thetas, X[i,:]) - Y[i]
    return X[i, :].T.dot(error)

def max_min_normalization(X):
    for i in range(1, int(X.shape[1])):
        max_value, min_value = np.max(X[:,i]), np.min(X[:,i])
        X[:,i] = (X[:,i]- min_value)/(max_value - min_value + 1e-10)
    return X

def z_score_normalization(X):
    for i in range(1, int(X.shape[1])):
        std, mean = np.std(X[:,i], ddof = 1), np.mean(X[:,i])
        X[:,i] = (X[:,i]- mean)/(std + 1e-10)
    return X

def GD(X, Y, thetas, rate = 0.1, max_iter = 100, end_condition=1e-4):
    X = z_score_normalization(X)
    sum_old, sum_new = 0, -mle(X, Y, thetas)
    iteration = 0
    while iteration < max_iter and abs(sum_old-sum_new) > end_condition:
        grads = X.T.dot(hypothesis(thetas, X) - Y)
        thetas -= rate * grads
        iteration += 1
        sum_old = sum_new
        sum_new = -mle(X, Y, thetas)
        show_plot(sum_new, X, Y, thetas, iteration)
        print("iteration%d:The hypothesis is y=%f + %f*x1 + %f*x2  Cost:%f" % (iteration, thetas[0], thetas[1], thetas[2], sum_new))

def SGD(X, Y, thetas, rate = 0.1, max_iter = 20, end_condition=1e-10):
    X = z_score_normalization(X)
    sum_old, sum_new = 0, -mle(X, Y, thetas)
    iteration = 0
    while iteration < max_iter and abs(sum_old-sum_new) > end_condition:
        show_plot(sum_new, X, Y, thetas, iteration)
        index_list = np.random.permutation(len(X))
        for index in index_list:
        #for index in range(len(X)):
            #index = random.randint(0, len(X)-1)
            grads = get_gradient(X, Y, thetas, index)
            thetas -= rate * grads
        iteration += 1
        sum_old = sum_new
        sum_new = -mle(X, Y, thetas)
        print("iteration %d:The hypothesis is y=%f + %f*x1 + %f*x2  Cost:%f" % (iteration, thetas[0], thetas[1], thetas[2], sum_new))

def min_SGD(X, Y, thetas, rate = 0.1, max_iter = 20, scale = 10, end_condition=1e-4):
    X = z_score_normalization(X)
    sum_old, sum_new = 0, -mle(X, Y, thetas)
    iteration = 0
    while iteration < max_iter:
        show_plot(sum_new, X, Y, thetas, iteration)
        index_list, rounds = np.random.permutation(len(X)), 0
        while rounds < len(X):
            if rounds + scale < len(X):
                indices = index_list[rounds:rounds+10]
                rounds +=scale
            else:
                indices = index_list[rounds:]
                rounds = len(X)
            grads = X[indices].T.dot(hypothesis(thetas, X[indices]) - Y[indices])
            thetas -= rate * grads
        iteration += 1
        sum_new = -mle(X, Y, thetas)
        print("iteration%d:The hypothesis is y=%f + %f*x1 + %f*x2  Cost:%f" % (iteration, thetas[0], thetas[1], thetas[2], sum_new))


def Netown(X, Y, thetas, end_condition=1e-4):
    X = z_score_normalization(X)
    sum_old, sum_new = 0, -mle(X, Y, thetas)
    iteration = 0
    while abs(sum_old-sum_new) > end_condition:
        show_plot(sum_new, X, Y, thetas, iteration)
        h = hypothesis(thetas, X)
        grads = np.dot(X.T, h-Y)
        H = X.T.dot(np.diag(h)).dot(np.diag(1-h)).dot(X)
        thetas -= np.linalg.inv(H).dot(grads)
        iteration += 1
        sum_old = sum_new
        sum_new = -mle(X, Y, thetas)
        print("iteration%d:The hypothesis is y=%f + %f*x1 + %f*x2  Cost:%f" % (iteration, thetas[0], thetas[1], thetas[2], sum_new))
        
def show_plot(sum_J, X, Y, thetas, iteration):
    exam2 = (-X[:,:-1].dot(thetas[:-1])/ (thetas[-1] + 1e-10)).tolist()
    X_0, X_1 = [], []
    for i in range(len(X)):
        X_1.append(X[i]) if Y[i]>0 else X_0.append(X[i]) 

    axarr[0].cla()
    axarr[0].plot([x[1] for x in X], exam2, 'r', color='red')
    axarr[0].plot([x[1] for x in X_0], [x[2] for x in X_0], '*', color='red')
    axarr[0].plot([x[1] for x in X_1], [x[2] for x in X_1], '+', color='blue')
    axarr[0].set_xlim([-2, 2])
    axarr[0].set_ylim([-2, 2])
    axarr[0].set_title('Logistic Regression with Gradient Descent Optimization')
    axarr[0].set_xlabel('Exam1')
    axarr[0].set_ylabel('Exam2')
    axarr[1].set_title('The convergence curve of cost function')
    axarr[1].scatter(iteration, sum_J, color='r')
    axarr[1].set_xlabel('Iteration steps')
    axarr[1].set_ylabel('Cost value')
    axarr[1].set_xlim([0, 20])
    axarr[1].set_ylim([30, 60])
    fig.canvas.draw()
    fig.show()

if __name__ == '__main__':
    fig, axarr = plt.subplots(1, 2, figsize=(14,5))
    X, Y = load_X_file('data\X.dat'), load_Y_file('data\Y.dat')
    thetas = [0, 0, 0]
    args = sys.argv[1:]
    if args[0]=="-h":
        print '''
               -h    -> help
               -f ['netown', 'gd', 'sgd', 'min_sgd']
                     -> optimization method
                e.g. python logist_demo.py -f gd
        '''
    elif args[0]=='-f':
        if args[1] == 'netown':
            Netown(X, Y, thetas)
        elif args[1]=='gd':
            GD(X, Y, thetas, 0.22)
        elif args[1]=='sgd':
            SGD(X, Y, thetas, 0.1, 20)
        elif args[1]=='min_sgd':
            min_SGD(X, Y, thetas, 0.1)
        else:
            print '''
                    -h    -> help
                    -f -> optimization method
                          ['netown', 'gd', 'sgd', 'min_sgd']
                    e.g. python logist_demo.py -f gd
                  ''' 
    raw_input()