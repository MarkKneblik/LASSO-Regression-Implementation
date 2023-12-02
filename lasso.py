import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def update(weights, bias, alpha, lambda_value, X_tr, Y_tr):
    #loss function for lasso regression: 1/m[sum((y_i - y_hat_i)^2) + lambda*sum(weights)] where m is number of data points

    Y_pred = np.dot(X_tr, weights) + bias # get prediction by taking dot product of X_tr and weight vector plus bias
   
    residual = Y_tr - Y_pred

    for i in range(len(weights)): # for each weight

        if weights[i] > 0: # if weight is positive
            weight_gradient = -2 * X_tr[i] * residual + lambda_value # add lambda so the abolute value of the gradient is bigger, pulling weight closer to 0 when updating

        else:
            weight_gradient = -2 * X_tr[i] * residual - lambda_value # else subtract lambda so the absolute value of the gradient is smaller, pulling it closer to 0 when updating

        weights[i] = weights[i] - alpha * weight_gradient # update weight by subtracting learning rate times gradient

    bias_gradient = -2 * residual # calculate gradient of bias
    bias = bias - alpha * bias_gradient # update bias

    return weights, bias


def gradient_descent(X_train, Y_train, alpha, lambda_value, n_iterations):
    d = X_train.shape[1] # feature dimension
    weights, bias = np.zeros(d), 0 # set weights and bias to zero

    for n in range(n_iterations):  # for each iteration of gradient descent
        # perform a full scan over the training dataset & update the weights and bias whenever the prediction is incorrect
        for example in range(len(X_train)): # for each training example
            X_tr = X_train[example] # X_tr is the training input associated with the current index of the X_train numpy array
            Y_tr = Y_train[example] # Y_tr is the training output associated with the current index of the Y_train numpy array
            weights, bias= update(weights, bias, alpha, lambda_value, X_tr, Y_tr) # get a tuple from update, which is in the form (weights[], bias)
            
    return weights, bias 


def predict(weights, bias, x_tst): # predict the label of x_tst
    y_pred = 0 # y_pred will hold dot product of x_tst and weights
    for i in range(len(weights)):
        y_pred += weights[i] * x_tst[i] # for every feature, multiply its value by its weight and add it to y_pred
    y_pred += bias # now y_pred = wTx, so add b to it

    return y_pred # replace this with the return of the prediction


def evaluate(weights, bias, X_test, Y_test):
    predictions = [] # hold all predictions
    for i in range(len(X_test)): # for every test example
        Y_pred = predict(weights, bias, X_test[i]) # predict value of Y at data point X_test[i]
        predictions.append(Y_pred)
        
    mse = mean_squared_error(Y_test, predictions) # calculate MSE of all predictions
    return mse


####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

df = pd.read_csv('Anchorage.csv', header=None)
df[1] = pd.to_datetime(df[1], errors='coerce') # convert date column to datetime type
df['Month'] = df[1].dt.month # create new column containing the month of that row (this will be 3.0 for March for example)
one_hot_encoded = pd.get_dummies(df['Month'], prefix='Month',dtype=float) # create new dataframe that contains dummy/indicator variables (one hot encoded months)
df = pd.concat([df, one_hot_encoded], axis=1) # concantenate the original dataframe and the one_hot_encoded dataframe so that we have 12 new columns in the old dataframe
df = df[1:] # exclude column names


df = df.drop(df.columns[[0,1,4,9,14,15,16,17,19,20,22]], axis = 1) # drop station and date columns, and drop additional columns that contain predominantly null values
df = df.drop('Month', axis = 1) # drop the month column that we created when one hot encoding
df.reset_index(drop=True, inplace=True) # reset row indices after dropping
df.columns = range(len(df.columns))  # reset column indices


df = df.to_numpy() # convert dataframe to numpy array


Y = df[:, 6] # target variable is the 7th column (columns start at 0)
X = np.hstack((df[:,:6], df[:, 7:])) # slice the df on the left and right of the target variable and stack them into one np


X = X.astype(float) # convert features into floats
Y = Y.astype(float) # targets into floats

# find rows that contain NaN and replace their values with the mean of the current column
for i in range(X.shape[1]):
    nan_indices = np.isnan(X[:, i])
    mean_value = np.nanmean(X[:, i])
    X[nan_indices, i] = mean_value


####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 80/20 train test split
lambda_value = 0 # initial lambda is 0 when testing for optimal alpha
n_iterations = 2 # number of iterations of gradient descent initially when testing alpha and lambda values

alpha_mse_list = [] # will hold mse value for corresponding alpha value
alpha_list = [] # list of alphas to test 

for i in range(0, 40): # testing 40 alpha values
    alpha = 0.00000001 + (0.00000001*i)    # use indexing to create alpha value to test
    alpha_list.append(alpha)
    weights, bias = gradient_descent(X_train, Y_train, alpha, lambda_value, n_iterations) # perform gradient descent
    mse = evaluate(weights, bias, X_test, Y_test) # calculate mse
    alpha_mse_list.append(mse) # add to mse list
    print(f'Mean Squared Error for alpha of {alpha}: {mse}')

# create plot for alpha values vs MSE
plt.figure()
plt.plot(alpha_list, alpha_mse_list, marker='o')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Alpha vs Mean Squared Error')
plt.show()

 
best_alpha_index = np.argmin(alpha_mse_list) # find index of alpha with smallest mse
best_alpha = alpha_list[best_alpha_index] # we will test variations of lambda with the best alpha value found
lambda_mse_list = [] # holds mse values for corresponding lambda values
lambda_list = [] # list of lambda values to be tested
step_size = 0.1 # used to create range of lambda values we will test

for lambda_value in np.arange(-8, -1.9, step_size): # from lambda of -8 to lambda of -1.9 in increments of 0.1
    lambda_list.append(lambda_value)
    weights, bias = gradient_descent(X_train, Y_train, alpha, lambda_value, n_iterations) # perform gradient descent with given values
    mse = evaluate(weights, bias, X_test, Y_test) # calculate mse
    lambda_mse_list.append(mse) # add to mse list for plotting
    print(f'Mean Squared Error for lambda of {lambda_value}: {mse}')

# create plot for lambda values vs MSE
plt.figure()
plt.plot(lambda_list, lambda_mse_list, marker='o')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.title('Lambda vs Mean Squared Error')
plt.show()

best_lambda_index = np.argmin(lambda_mse_list) # find index of lambda with smallest mse value
best_lambda = lambda_list[best_lambda_index] # use index to find the best lambda value

final_mse_list = [] # will hold lists of MSE values for each number of iterations (each list will hold 5 mse values because we are testing each # of iterations 5 times)
iterations_list = np.arange(1, 26) # create sequence of numbers of iterations from 1 to 25

for n_tests in range(5): # 5 runs for each number of iterations
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # 80/20 train/test split
    mse_per_run = [] # list that holds 25 mse values for 1-25 iterations

    for n_iterations in iterations_list: 
        print(f"Test {n_tests + 1}, Iterations: {n_iterations}")
        weights, bias = gradient_descent(X_train, Y_train, best_alpha, best_lambda, n_iterations) # perform gradient descent with given parameters
        mse = evaluate(weights, bias, X_test, Y_test) # calculate mse
        mse_per_run.append(mse) # append to list that holds all mse values for this number of iterations
        print(f'Mean Squared Error for {n_iterations} iterations: {mse}')

    final_mse_list.append(mse_per_run) # after every number of iterations is tested, add this list to the final_mse_list

average_mse_list = np.mean(final_mse_list, axis=0) # find average MSE for each number of iterations

# create plot for number of iterations of gradient descent vs MSE
plt.figure()
plt.plot(iterations_list, average_mse_list, marker='o')
plt.xlabel('No. of Iterations')
plt.ylabel('Average Mean Squared Error')
plt.title('No. of Iterations vs Average Mean Squared Error')
plt.show()