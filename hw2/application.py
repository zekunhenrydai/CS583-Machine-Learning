import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE
w = train(Xtrain, Ytrain, alpha=0.1, n_epoch = 500)
yhat = Xtrain.dot(w)
yhat_test = Xtest.dot(w)
L = compute_L(yhat, Ytrain)
L_test = compute_L(yhat_test, Ytest)
assert np.allclose(L, 0., atol = 1e-2)
assert np.allclose(L_test, 0., atol = 1e-2)
print("Loss on training set:")
print(L)
print("Loss on training set is less than 1e-2:")
print(L<1e-2)
print( )
print("Loss on testing set:")
print(L_test)
print("Loss on testing set is less than 1e-2:")
print(L_test<1e-2)

loss_test = []
al = []
alpha = 0.015
while alpha < 0.1:
    al.append(alpha)
    w = train(Xtrain, Ytrain, alpha, n_epoch = 500)
    yhat = Xtrain.dot(w)
    yhat_test = Xtest.dot(w)
    L = compute_L(yhat, Ytrain)
    L_test = compute_L(yhat_test, Ytest)
    loss_test.append(L_test)
    alpha += 0.0001
print(loss_test)
import matplotlib.pyplot as plt
plt.xlabel("Alpha value")
plt.ylabel("Loss on test set")
plt.plot(al, loss_test, color="r")
plt.show()

# loss_test = []
# epoch = []
# for n_epoch in range(500, 1000):
#     epoch.append(n_epoch)
#     w = train(Xtrain, Ytrain, alpha=0.01, n_epoch=n_epoch)
#     yhat = Xtrain.dot(w)
#     yhat_test = Xtest.dot(w)
#     L = compute_L(yhat, Ytrain)
#     L_test = compute_L(yhat_test, Ytest)
#     loss_test.append(L_test)
# import matplotlib.pyplot as plt
# plt.xlabel("Epochs")
# plt.ylabel("Loss on test set")
# plt.plot(epoch, loss_test, color="b")
# plt.show()
#########################################

