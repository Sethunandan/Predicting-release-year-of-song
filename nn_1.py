import sys
import os
import numpy as np
import pandas as pd

np.random.seed(42)

def relu(w):
    return np.clip(w, 0, None)



NUM_FEATS = 90

class Net(object):
    '''
    '''

    def __init__(self, num_layers, num_units):
        '''
        Initialize the neural network.
        Create weights and biases.
        Here, we have provided an example structure for the weights and biases.
        It is a list of weight and bias matrices, in which, the
        dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
        weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
        biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]
        Please note that this is just an example.
        You are free to modify or entirely ignore this initialization as per your need.
        Also you can add more state-tracking variables that might be useful to compute
        the gradients efficiently.
        Parameters
        ----------
            num_layers : Number of HIDDEN layers.
            num_units : Number of units in each Hidden layer.
        '''
        self.num_layers = num_layers
        self.num_units = num_units

        self.biases = []
        self.weights = []
        for i in range(num_layers):

            if i==0:
                # Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
                

            else:
                # Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))


            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))


        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
        

    def __call__(self, X):
        '''
        Forward propagate the input X through the network,
        and return the output.
        
        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
        Returns
        ----------
            y : Output of the network, numpy array of shape m x 1
            
        '''
        
        self.h_states = []
        self.a_states = []
        a = X


        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i==0:
                self.h_states.append(a) # For input layer, both h and a are same
            else:
                self.h_states.append(h)
            self.a_states.append(a)

            h = np.dot(a, w) + b.T

            if i < len(self.weights)-1:
                a = relu(h)
            else: # No activation for the output layer
                a = h

        self.pred = a
        
        return self.pred
    
        raise NotImplementedError

    def backward(self, X, y, lamda):
        '''
        Compute and return gradients loss with respect to weights and biases.
        (dL/dW and dL/db)
        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
            y : Output of the network, numpy array of shape m x 1
            lamda : Regularization parameter.
        Returns
        ----------
            del_W : derivative of loss w.r.t. all weight values (a list of matrices).
            del_b : derivative of loss w.r.t. all bias values (a list of vectors).
        Hint: You need to do a forward pass before performing bacward pass.
        '''
        del_W = []
        del_b = []
        dA = 2*(self.pred - y)
        
        for  i in reversed(range(self.num_layers+1)):
            m = y.shape[0]
            

            dZ = dA
                

                
            
            dW = np.dot(dZ.T,self.a_states[i])/m
            dW = dW.T + lamda * self.weights[i]
            del_W.append(dW)
            
            db = np.sum(dZ.T, axis=1, keepdims=True)/m + lamda * self.biases[i]
            del_b.append(db)
            
            
            dA_prev = np.dot(self.weights[i],dZ.T)
            dA = dA_prev.T
    
        del_W.reverse()
        del_b.reverse()
        
        return del_W, del_b
        
        raise NotImplementedError

class Optimizer(object):
    '''
    '''

    def __init__(self, learning_rate=None, num_layers=None, num_units=None):

        
        '''
        Create a Gradient Descent based optimizer with given
        learning rate.
        Other parameters can also be passed to create different types of
        optimizers.
        Hint: You can use the class members to track various states of the
        optimizer.
        '''
        
        self.l = learning_rate
        self.num_layers = num_layers
        self.num_units = num_units
        self.v={}
        self.m={}
        self.mhat={}
        self.vhat={}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        
        for i in range(num_layers+1):

            if i==0:
                # Input layer
                self.m['dW'+str(i)]=np.zeros((NUM_FEATS, self.num_units))
                self.v['dW'+str(i)]=np.zeros((NUM_FEATS, self.num_units))
                
            elif i==num_layers:
                # Output layer
                self.m['dW'+str(i)]=np.zeros((self.num_units, 1))
                self.v['dW'+str(i)]=np.zeros((self.num_units, 1))
            
            else:
                # Hidden layer
                self.m['dW'+str(i)]=np.zeros((self.num_units, self.num_units))
                self.v['dW'+str(i)]=np.zeros((self.num_units, self.num_units))
            
            if i != num_layers:
                self.m['db'+str(i)]=np.zeros((self.num_units, 1))
                self.v['db'+str(i)]=np.zeros((self.num_units, 1))
            else:
                # Output layer
                self.m['db'+str(i)]=np.zeros((1,1))
                self.v['db'+str(i)]=np.zeros((1,1))
        
        

    def step(self, weights=None, biases=None, delta_weights=None, delta_biases=None, epoch=None):

        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.
        '''
        for i in range(len(weights)):
            
            self.m['dW'+str(i)] = self.beta1*self.m['dW'+str(i)]+(1-self.beta1)*delta_weights[i]
            self.v['dW'+str(i)] = self.beta2*self.v['dW'+str(i)]+(1-self.beta2)*(delta_weights[i]*delta_weights[i])
            
            self.m['db'+str(i)] = self.beta1*self.m['db'+str(i)]+(1-self.beta1)*delta_biases[i]
            self.v['db'+str(i)] = self.beta2*self.v['db'+str(i)]+(1-self.beta2)*(delta_biases[i]*delta_biases[i])
            
            self.mhat['dW'+str(i)] = self.m['dW'+str(i)]/(1-np.power(self.beta1, epoch+1))
            self.vhat['dW'+str(i)] = self.v['dW'+str(i)]/(1-np.power(self.beta2, epoch+1))
            
            self.mhat['db'+str(i)] = self.m['db'+str(i)]/(1-np.power(self.beta1, epoch+1))
            self.vhat['db'+str(i)] = self.v['db'+str(i)]/(1-np.power(self.beta2, epoch+1))
            
            weights[i] = weights[i] - (self.l * self.mhat['dW'+str(i)])/(np.sqrt(self.vhat['dW'+str(i)])+self.eps)
            biases[i] = biases[i] - (self.l * self.mhat['db'+str(i)])/(np.sqrt(self.vhat['db'+str(i)])+self.eps)
            

            
        return weights,biases
            
        raise NotImplementedError

def loss_mse(y, y_hat):
    '''
    Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
        MSE loss between y and y_hat.
    '''
    loss = np.dot((y_hat-y).T,y_hat-y)/y.shape[0]
    
    return loss[0][0]
    
    raise NotImplementedError
    
    
def loss_regularization(weights, biases):
    '''
    Compute l2 regularization loss.

    Parameters
    ----------
        weights and biases of the network.

    Returns
    ----------
        l2 regularization loss 
        
    '''
    w_sum= 0
    b_sum = 0
    for i in zip(weights,biases):
        w_sum += np.linalg.norm(i[0],2)
        b_sum += np.linalg.norm(i[1],2)
    
    return (w_sum+b_sum)    
    
    raise NotImplementedError
    
def loss_fn(y, y_hat, weights, biases, lamda):
    '''
    Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1
        weights and biases of the network
        lamda: Regularization parameter

    Returns
    ----------
        l2 regularization loss 
    '''
    return lamda*loss_regularization(weights, biases) + loss_mse(y, y_hat)
    
    
    raise NotImplementedError

def rmse(y, y_hat):
    '''
    Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
        RMSE between y and y_hat.
    '''
    return (np.sqrt(np.dot((y-y_hat).T,(y-y_hat))/y.shape[0]))[0][0]
    raise NotImplementedError

def train(
    net, optimizer, lamda, batch_size, max_epochs,
    train_input, train_target,
    dev_input, dev_target
):
    '''
    In this function, you will perform following steps:
        1. Run gradient descent algorithm for `max_epochs` epochs.
        2. For each bach of the training data
            1.1 Compute gradients
            1.2 Update weights and biases using step() of optimizer.
        3. Compute RMSE on dev data after running `max_epochs` epochs.
    Here we have added the code to loop over batches and perform backward pass
    for each batch in the loop.
    For this code also, you are free to heavily modify it.
    '''

    m = train_input.shape[0]

    for e in range(max_epochs):
        epoch_loss = 0.
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i+batch_size]
            batch_target = train_target[i:i+batch_size]
            pred = net(batch_input)

            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, lamda)

            # Get updated weights based on current weights and gradients
            weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db,e)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
            epoch_loss += batch_loss

#             print(e, i, rmse(batch_target, pred), batch_loss)

#         print(e, epoch_loss)

        # Write any early stopping conditions required (only for Part 2)
        # Hint: You can also compute dev_rmse here and use it in the early
        # 		stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    dev_pred = net(dev_input)
    train_pred = net(train_input)
    dev_rmse = rmse(dev_target, dev_pred)
    train_rmse = rmse(train_target,train_pred)
    
    return train_rmse, dev_rmse

    


def get_test_data_predictions(net, inputs):
    '''
    Perform forward pass on test data and get the final predictions that can
    be submitted on Kaggle.
    Write the final predictions to the part2.csv file.
    Parameters
    ----------
        net : trained neural network
        inputs : test input, numpy array of shape m x d
    Returns
    ----------
        predictions (optional): Predictions obtained from forward pass
                                on test data, numpy array of shape m x 1
    '''
    test_out = net(inputs)
    out_df = pd.DataFrame(test_out,columns = ['Predicted'])
    out_df['Id'] = range(1,test_out.shape[0]+1)
    out_df = out_df.astype({'Id' :float})
    out_df.set_index('Id',inplace = True)
    out_df.to_csv('submission.csv')
    return None
    raise NotImplementedError

def read_data(path, is_train = False):
    '''
    Read the train, dev, and test datasets
    '''
    data = pd.read_csv(path)
    
    if is_train:
        return np.array(data.iloc[:,1:]),np.array(data[['label']])
    else:
        return data
    
    raise NotImplementedError
    


def main():

    # These parameters should be fixed for Part 1
    max_epochs = 50
    batch_size = 128


    
    
    
    train_input, train_target = read_data("/kaggle/input/cs725-autumn-2020-programming-assignment-2/dataset/train.csv",True)
    dev_input, dev_target = read_data("/kaggle/input/cs725-autumn-2020-programming-assignment-2/dataset/dev.csv",True)
    test_input = read_data("/kaggle/input/cs725-autumn-2020-programming-assignment-2/dataset/test.csv")
    train_error = []
    dev_error  = []
    lr = []
    numlayer = []
    numunit = []
    ld = []
    
    for (learning_rate,num_layers,num_units,lamda) in [(i,j,k,l) for i in [0.001,0.01] for j in [1,2] for k in [64,128] for l in [0,5]]:

        net = Net(num_layers, num_units)
        optimizer = Optimizer(learning_rate,num_layers, num_units)
        train_rmse, dev_rmse = train(
            net, optimizer, lamda, batch_size, max_epochs,
            train_input, train_target,
            dev_input, dev_target
        )
        train_error.append(train_rmse)
        dev_error.append(dev_rmse)
        lr.append(learning_rate) 
        numlayer.append(num_layers)
        numunit.append(num_units)
        ld.append(lamda)
        
        
        
        print('learning rate = {} no. of hidden layers = {} Size of each hidden layer = {} lambda = {} train_rmse = {:.5f} dev_rmse = {:.5f}\n'.format(learning_rate,num_layers,num_units,lamda,train_rmse, dev_rmse))
        
    get_test_data_predictions(net, test_input)

    
    part1b = pd.DataFrame(lr,columns = ['Learning Rate'])
    part1b['No. of hidden layers'] = numlayer
    part1b['Size of each hidden layer'] = numunit
    part1b['regulariser'] = ld
    part1b['RMSE(train)'] = train_error
    part1b['RMSE(dev)'] = dev_error
    part1b.to_csv('part_1b.csv',index = False)
    


if __name__ == '__main__':
    main()
