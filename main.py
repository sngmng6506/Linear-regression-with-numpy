#dataset : https://www.kaggle.com/code/sudhirnl7/linear-regression-tutorial/notebook
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt

data = pd.read_csv('insurance.csv')
data_for_train = data[0:1200].to_numpy()
data_for_test = data[1200:].to_numpy()

## pre-processing
def Pre_proc(data):
    num_row_train = np.shape(data)[0]
    age = data[:,0]
    sex = []
    bmi = data[:,2]
    children = data[:,3]
    smoker = []
    region = []
    charges = data[:,6]
    for i in range(num_row_train):
        if data[i,1] == 'female':
            sex += [1]
        else:
            sex += [0]
    for i in range(num_row_train):
        if data[i,4] == 'yes':
            smoker += [1]
        else:
            smoker += [0]
    for i in range(num_row_train):
        if data[i,5] == 'southwest':
            region += [1]
        else:
            region += [0]

    sex = np.array(sex)
    smoker = np.array(smoker)
    region = np.array(region)

    # MIN-MAX normalize
    age =np.array(( age - np.min(age) ) / (np.max(age) - np.min(age)))
    bmi =np.array(( bmi - np.min(bmi) ) / (np.max(bmi) - np.min(bmi)))
    children =np.array(( children - np.min(children) ) / (np.max(children) - np.min(children)))
    charges =np.array(( charges - np.min(charges) ) / (np.max(charges) - np.min(charges)))

    X = np.stack((age.T,sex.T,bmi.T,children.T,smoker.T,region.T),axis = 1)
    Y = charges
    return X,Y

#####################
# analytic solution
#####################
X,Y = Pre_proc(data_for_train)

theta1 = np.linalg.inv(X.T.dot(X).astype(np.float32))
theta2 = X.T.dot(Y)
theta_anal = theta1.dot(theta2)
hypothesis_anal = X.dot(theta_anal)

cost_anal = np.sum(np.square(hypothesis_anal - Y ))  / Y.shape[0]


####################
# Gradient Descent
####################

def optimize_theta(theta,X,Y):
    #theta_init = np.ones([6,])
    theta_init = theta
    hypothesis_init = X.dot(theta_init)
    cost_init = np.sum(np.square(hypothesis_init - Y) / Y.shape[0])
    delta = 0.001

    theta_perturbed = np.zeros((6,6))
    for i in range(6):
        theta_perturbed[i] = theta_init
        theta_perturbed[i][i] -= delta*theta_init[i]


    hypothesis_perturbed = X.dot(theta_perturbed.T) # (1200,6)
    Y_for_perturbed = np.repeat(Y,6).reshape((1200,6))

    tmp = abs(Y_for_perturbed - hypothesis_perturbed)
    cost_perturbed = np.zeros((6,)) ##(6,)
    for i in range(6):
        a = np.sum(np.square(tmp[:,i]))/Y.shape[0]
        cost_perturbed[i] = a

    ## gradient cost

    gradient_cost = ( cost_init - cost_perturbed ) / delta

    alpha = 0.4 #learning rate
    theta = theta_init - alpha * gradient_cost

    hypothesis_next = X.dot(theta)
    cost_next = np.sum(np.square(hypothesis_next - Y) / Y.shape[0])

    #print('cost = {}'.format(cost_next))
    return theta, cost_next
# iter to optimize the theta

theta_op = np.ones([6,]) #inital condition
for i in tqdm.tqdm(range(4000)):
    theta_op,cost_op = optimize_theta(theta_op,X,Y)




print("analytic theta = {} , analytic cost = {}".format(theta_anal,cost_anal))
print("G.D theta = {} , G,D cost = {}".format(theta_op,cost_op))



#test
X,Y = Pre_proc(data_for_test)

hypothesis = X.dot(theta_op)
RMS = np.sum(np.square(hypothesis-Y))/len(Y)
print("RMS in test  = {}".format(RMS))











