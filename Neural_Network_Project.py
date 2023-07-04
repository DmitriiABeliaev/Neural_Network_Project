#!/usr/bin/env python
# coding: utf-8

# # Group Project: Machine Learning 
# 
# - <a href='#part1'>Part 1: Deep L-Layer Neural Network for Image Classification</a>
# 	- You will use pre-built functions to build an L-Layer neural network for an image classification task
# - <a href='#part2'>Part 2: Full Machine Learning Project</a>
# 	- You will go through the full "idea, code, experiment" cycle to build and improve a model of your choice
# 
# You may work in groups of 1-3 students for this project.
# 
# In this project, especially in Part 2, you are expected to show the work you have done in the form of including results for models you have experimented with on the path to the best-performing model. Make sure you include Python and markdown boxes explaining and discussing any decisions you have made and interpretations of the results you have achieved. You can include diagrams, tables, and/or graphs using markdown. **A significant portion of your grade will be based on the progression of your model, not just the final result.**
# 
# **Note**: All work you submit must be the work of your group. Projects will be checked against each other, and against any work submitted in previous semesters where a similar project was given.

# <a id="part1"></a>
# ## Part 1: Deep L-Layer Neural Network for Image Classification
# 
# You will use the functions given to you to build a deep L-layer network, and apply it to cat vs non-cat classification. Hopefully, you will see an improvement in accuracy relative to your previous logistic regression implementation.  

# ### 1.1 - Packages

# Let's first import all the packages that you will need during this assignment. 
# - [numpy](https://www.numpy.org/) is the fundamental package for scientific computing with Python.
# - [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.
# - [h5py](http://www.h5py.org) is a common package to interact with a dataset that is stored on an H5 file.
# - [PIL](https://pillow.readthedocs.io/en/stable/index.html) and [scipy](https://www.scipy.org/) are used here to test your model with your own picture at the end.
# - nn_functions provides the functions you need to build an L-layer network.
# - np.random.seed(1) is used to keep all the random function calls consistent. It will help us grade your work.

# In[1]:


import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from nn_functions import *
from extra_functions import *

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)


# ### 1.2 - Dataset
# 
# You will use the same "Cat vs non-Cat" dataset as in your previous assignment. The model you had built had 70% test accuracy on classifying cats vs non-cats images. Hopefully, your new model will perform a better!
# 
# **Problem Statement**: You are given a dataset containing:
# - a training set of m_train images labelled as cat (1) or non-cat (0)
# - a test set of m_test images labelled as cat and non-cat
# - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
# 
# Let's get more familiar with the dataset. Load the data by running the cell below.

# In[2]:


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# The following code will show you an image in the dataset. Feel free to change the index and re-run the cell multiple times to see other images. 

# In[3]:


# Example of a picture
index = 50
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")


# In[4]:


# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


# As usual, you reshape and standardize the images before feeding them to the network. The code is given in the cell below.
# 
# <img src="images/imvectorkiank.png" style="width:450px;height:300px;">
# 
# <caption><center> <u>Figure 1</u>: Image to vector conversion. <br> </center></caption>

# In[5]:


# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


# $12,288$ equals $64 \times 64 \times 3$ which is the size of one reshaped image vector.

# ### 1.3 - Architecture of your model

# Now that you are familiar with the dataset, it is time to build a deep neural network to distinguish cat images from non-cat images.
# 
# Here is a simplified network representation for an L-layer neural network:
# 
# <img src="images/LlayerNN_kiank.png" style="width:650px;height:400px;">
# <caption><center> <u>Figure 2</u>: L-layer neural network.</center></caption> 
# 
# The model can be summarized as: ***[LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID***</center></caption>
# 
# <u>Detailed Architecture of figure 2</u>:
# - The input is a (64,64,3) image which is flattened to a vector of size (12288,1).
# - The corresponding vector: $[x_0,x_1,...,x_{12287}]^T$ is then multiplied by the weight matrix $W^{[1]}$ and then you add the intercept $b^{[1]}$. The result is called the linear unit.
# - Next, you take the relu of the linear unit. This process could be repeated several times for each $(W^{[l]}, b^{[l]})$ depending on the model architecture.
# - Finally, you take the sigmoid of the final linear unit. If it is greater than 0.5, you classify it to be a cat.
# 
# <u>General methodology</u>
# 
# As usual you will follow the Deep Learning methodology to build the model:
# 1. Initialize parameters / Define hyperparameters
# 2. Loop for num_iterations:
#     - Forward propagation
#     - Compute cost function
#     - Backward propagation
#     - Update parameters (using parameters, and grads from backprop) 
# 4. Use trained parameters to predict labels

# ### 1.4 - L-layer Neural Network
# 
# **Exercise**: Use the helper functions in the nn_functions file to build an $L$-layer neural network with the following structure: **[LINEAR -> RELU]$\times$(L-1) -> LINEAR -> SIGMOID**. Spend some time looking through the functions and understanding how they can be used to build a deep neural network. The functions you may need and their inputs are:
# ```python
# def initialize_parameters_deep(layers_dims):
#     ...
#     return parameters 
# def L_model_forward(X, parameters):
#     ...
#     return AL, caches
# def compute_cost(AL, Y):
#     ...
#     return cost
# def L_model_backward(AL, Y, caches):
#     ...
#     return grads
# def update_parameters(parameters, grads, learning_rate):
#     ...
#     return parameters
# ```

# In[6]:


### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


# In[7]:


# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, cache = L_model_forward(X, parameters)

        # Compute cost.
        cost = compute_cost(AL, Y)
        
        # Backward propagation.
        
        grads = L_model_backward(AL, Y, cache)
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# You will now train the model as a 4-layer neural network. 
# 
# Run the cell below to train your model. The cost should decrease on every iteration. It may take a few minutes to run 2500 iterations. 

# In[8]:


parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


# In[9]:


pred_train = predict(train_x, train_y, parameters)


# In[10]:


pred_test = predict(test_x, test_y, parameters)


# Congratulations! It seems that your 4-layer neural network has better performance than your previous assignment network on the same test set. 
# 
# This is good performance for this task.

# ###  1.5 - Results Analysis
# 
# First, let's take a look at some images the L-layer model labeled incorrectly. This will show a few mislabeled images. 

# In[11]:


print_mislabeled_images(classes, test_x, test_y, pred_test)


# **A few types of images the model tends to do poorly on include:** 
# - Cat body in an unusual position
# - Cat appears against a background of a similar color
# - Unusual cat color and species
# - Camera Angle
# - Brightness of the picture
# - Scale variation (cat is very large or small in image) 

# ### 1.6 - Test with your own image (optional/ungraded exercise) ##
# 
# You can use your own image and see the output of your model. To do that:
# 1. Add your image to the "images" folder
# 2. Change your image's name in the following code
# 3. Run the code and check if the algorithm is right (1 = cat, 0 = non-cat)!

# In[12]:


my_image = "cat.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)


fname = "images/" + my_image
image = Image.open(fname)
my_image = np.array(image.resize((num_px, num_px))).reshape((1, num_px*num_px*3)).T
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


# <a id="part2"></a>
# ## Part 2: Full Machine Learning Project
# 
# Now you will use all you know about building and training neural networks in an "idea, code, experiment" cycle on a data set.
# 
# ### 2.1 - Find a dataset (or datasets)
# 
# Find an appropriate dataset to work with. Some places to look:
# - [Kaggle](https://www.kaggle.com/datasets) 
# - [University of California, Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)
# 
# You can choose just one, or more than one if you'd like to. Think carefully of the type of task you are trying to accomplish (e.g., classification, regression, etc.). Spend some time analyzing and processing the data. For example, decide how to split the data; should you have separate train, dev, and test sets? Does the data need to be cleaned or adjusted? How should the data be normalized? Any other considerations or adjustments needed for the data?
# 
# Clearly indicate where you found the dataset(s) you are working with.
# 
# Show the work you have done analyzing and processing the data in Python boxes in this notebook. There should also be associated markdown boxes discussing what you have observed and what decisions you have made.
# 
# 
# 

# ## DEALING WITH DIFFERENT TYPES OF DATASET 
# 
# We tried different types of dataset to make them fit with our model. But there was a dataset that had 8 attributes and 3 different outputs, we tried al ot to make the model provided in part-1 work on that dataset. Ashtonishing observtions were made as they was no correlation and the cost computed was in negative and with increase in number of iterations the cost started increasing.
# 
# ## OUR DATASET 
# 
# The dataset chosen has 11 attributes and 1 output column. This dataset predicts that if the patient has the heart disease or not considering different factors that affect the prediction.

# In[2]:


# Your work for analyzing and processing the data
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pandas as pd


# We decided to change all text inputs into numbers for our cost calculations. All of our text data happened to be categories so we assigned each category to a number so that it would be easier for the model to compute. 

# In[3]:


df = pd.read_csv("heart.csv")

df['Sex'] = df['Sex'].replace('M', 1).replace('F', 2)
df['ChestPainType'] = df['ChestPainType'].replace('ATA', 1).replace('NAP', 2).replace('ASY', 3).replace('TA', 4)
df['RestingECG'] = df['RestingECG'].replace('Normal', 1).replace('ST', 2).replace('LVH', 3)
df['ExerciseAngina'] = df['ExerciseAngina'].replace('N', 1).replace('Y', 2)
df['ST_Slope'] = df['ST_Slope'].replace('Up', 1).replace('Flat', 2).replace('Down', 3)
dataSet = df.to_numpy()


# We split the train and test data to a 90/10 split. This choice allowed us more data to use for training. Then we normalized the data including the categories which we replaced with integers such that the values in each row add up to 1. 

# In[4]:


X = dataSet[:, :-1]
Y = dataSet[:, -1:]
X_normed = normalize(X, axis=1, norm='l1')

X_train, X_test, y_train, y_test = train_test_split(X_normed, Y, test_size=0.1)


# ### 2.2 - Build your model
# 
# Start with a basic model, show the results, and then apply whichever improvements you decide to incorporate as per below.
# 
# You have two options for building your model:
# - **The difficult option**: Use the provided L-layer network code used above in Part 1 and (later) extend it to incorporate more advanced neural network improvements as given in class
# - **The easier option**: Use [Keras](https://keras.io) and [TensorFlow](https://www.tensorflow.org) to build a network
# 	- You may *not* use any framework other than Keras/TensorFlow
# 
# Ambition will be rewarded! If you choose the easier option, you are expected to incorporate more of the potential improvements given below.
# 
# Some of the neural network improvements you can consider incorporating for either option (not an exhaustive list):
# - Weight initialization methods (e.g., zeroes, random, etc.)
# - Regularization: L2, dropout, etc.
# - Mini-batch gradient descent
# - Gradient descent optimization algorithm: momentum, RMSProp, Adam, etc.
# - Batch normalization
# 
# Show the results with your model with improvements. Use markdown boxes to discuss the effect of your improvement(s) and change in accuracy.

# ## Explanations ##
# 
# In this part we actually implement the code for our model on the selected dataset and see the results. 
# Here we compare 4 different scenarios
# 
# -  Plane model with no improvements
# -  Model with L2 regularization
# -  Model with gradient descent with momentum optimization
# -  Model with both the above mentioned improvements
# 
# In our selected dataset we found out that we don't have much variance so, applying improvements like L2 regularization does not drastically improve the model but it makes some little improvements.
# 
# Our dataset is not very huge so we decided to not to implement mini batch optimization as that would have been ineffective on such a dataset.

# In[5]:


# Your work for building a basic model and then applying improvements

n_x = X_train.shape[1]
n_h = 4
n_y = X_train.shape[1]
newy_train = y_train.T

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, beta = 0.9, lambd = 0.5):
    
    costs = []
    newy_train = Y.T
    X_train = X
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        
        AL, cache = L_model_forward(X_train.T, parameters)

        # Compute cost & Backward propagation.
        
        cost = compute_cost(AL, newy_train)
        grads = L_model_backward(AL, newy_train, cache)
    
        # Update parameters.
        
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    print("Original Model")
    # plot the cost
    plt.figure(figsize=(5,5))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[6]:



def L_layer_model_L2(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, beta = 0.9, lambd = 0.5):

    costs = []
    newy_train = Y.T
    X_train = X
    
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, cache = L_model_forward(X_train.T, parameters)

        # Compute cost & Backward propagation.
        cost = L2_regularization(AL, newy_train, parameters, cache, lambd)
        grads = backward_regularization(AL, newy_train, cache, 1, lambd)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    print("Model with L2")
    # plot the cost
    plt.figure(figsize=(5,5))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[7]:



def L_layer_model_momentum(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, beta = 0.9, lambd = 0.5):

    costs = []
    newy_train = Y.T
    X_train = X
    parameters = initialize_parameters_deep2(layers_dims)
    
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, cache = L_model_forward2(X_train.T, parameters)
        
        # Compute cost & Backward propagation.
        cost = compute_cost(AL, newy_train)
        grads = L_model_backward(AL, newy_train, cache)
    
        # Update parameters.
        parameters = momentum_update(parameters, grads, learning_rate, beta)
       
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    print("Model with momentum")
    # plot the cost
    plt.figure(figsize=(5,5))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[8]:



def L_layer_model_L2_momentum(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, beta = 0.9, lambd = 0.5):

    costs = []
    newy_train = Y.T
    X_train = X
    parameters = initialize_parameters_deep2(layers_dims)
   
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        
        AL, cache = L_model_forward2(X_train.T, parameters)
        
        # Compute cost & Backward propagation.
        cost = L2_regularization(AL, newy_train, parameters, cache, lambd)
        grads = backward_regularization(AL, newy_train, cache, 1, lambd)
       
    
        # Update parameters.
        parameters = momentum_update(parameters, grads, learning_rate, beta)
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    print("Model with L2 and momentum")            
    # plot the cost
    plt.figure(figsize=(5,5))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# Cell below allows to print all variations of models and its accuracies with standart hyperparameters. To change the print out, we could just change the boolean values of **L2** and **Momentum** to either True or False

# In[12]:


layers_dims = [11, 20, 7, 5, 1]
num_iterations = 2500
learning_rate = 0.075 
beta = 0.9
lambd = 0.1
print_cost = True

################### CHANGE HERE ##########################

L2 = True
Momentum = True

################### CHANGE HERE ##########################

if(L2 == True and Momentum == True):
    parameters1 = L_layer_model_L2_momentum(X_train, y_train, layers_dims, learning_rate = learning_rate, num_iterations = num_iterations, print_cost = print_cost, beta = beta, lambd = lambd)
    print("Train Accuracy")
    pred_train = predict2_only_prob(X_train.T, y_train.T, parameters1) 
    print(pred_train)
    print("Test Accuracy")
    pred_test = predict2_only_prob(X_test.T, y_test.T, parameters1)
    print(pred_test)
if(L2 == True and Momentum == False):
    parameters2 = L_layer_model_L2(X_train, y_train, layers_dims, learning_rate = learning_rate, num_iterations = num_iterations, print_cost = print_cost, beta = beta, lambd = lambd)
    print("Train Accuracy")
    pred_train = predict_only_prob(X_train.T, y_train.T, parameters2) 
    print(pred_train)
    print("Test Accuracy")
    pred_test = predict_only_prob(X_test.T, y_test.T, parameters2)
    print(pred_test)
if(L2 == False and Momentum == True):
    parameters3 = L_layer_model_momentum(X_train, y_train, layers_dims, learning_rate = learning_rate, num_iterations = num_iterations, print_cost = print_cost, beta = beta, lambd = lambd)
    print("Train Accuracy")
    pred_train = predict2_only_prob(X_train.T, y_train.T, parameters3) 
    print(pred_train)
    print("Test Accuracy")
    pred_test = predict2_only_prob(X_test.T, y_test.T, parameters3)
    print(pred_test)
elif(L2 == False and Momentum == False):
    parameters4 = L_layer_model(X_train, y_train, layers_dims, learning_rate = learning_rate, num_iterations = num_iterations, print_cost = print_cost, beta = beta, lambd = lambd)
    print("Train Accuracy")
    pred_train = predict_only_prob(X_train.T, y_train.T, parameters4) 
    print(pred_train)
    print("Test Accuracy")
    pred_test = predict_only_prob(X_test.T, y_test.T, parameters4)
    print(pred_test)


# ### EACH OF THE POSSIBLE MODELS ### 
# 
# - **Original Model**: original model from part 1 without any improvement
# 
# <img src="images/stnd_org_model.png">
# 
# - **L2 Model**: model with L2 regularization which implements changes for cost and beckwards propagation calculations
# 
# <img src="images/stnd_L2_model.png">
# 
# - **Momentum Model**: model with momentum which implements main changes to forward propagation and parameters' updates
# 
# <img src="images/stnd_Moment_model.png">
# 
# - **L2 and Momentum Model**: final model with L2 regularization and momentum improvements
# 
# <img src="images/stnd_L2_Moment_model.png">
# 

# ### 2.3 - Idea, Code, Experiment Cycle
# 
# Now go through a iterative process to improve your model. This will involve things like (not an exhaustive list):
# - Checking whether you have a bias and/or variance problem. How will you address it? 
# - Hyperparameter tuning: learning rate, # of layers, # of hidden units, activation functions, mini-batch size, etc.
# - Trying any of the improvements made to the model in part 2.2 to see if it leads to better results
# 
# We want to see the progression of your model to a final version with the best results you can achieve. You don't have to show results for every single experiment you tried, but a general progression of different models at various stages of development should be included. Feel free to include discussion, diagrams, tables, and/or graphs that may summarize some of your experiments. **If you only show us the final model you've built, your mark will be minimal**.
# 
# Your discussion should also include details on the methodology you used in your experiments. For example, how did you approach hyperparameter tuning?
# 

# ### Explanation ###

# Initially we implemented L2 Regularization because our hypothesis was that we had high variance. After implementing L2 regularization, we found that the costs changed, but the test and train accuracies did not change much. When we adjusted the values for lambda, we did not see much of a improvement in the accuracy. This indicated to us that we did not have a high variance problem but we had high bias. So we increased the number of iterations on our training data which helped with accuracy. 
# 
# When we implemented L2 Regularization we noticed that the graph looked ragged, and since L2 Regularization didn't do much for our accuracy we decided to try smoothing it out using Gradient descent with momentum. We implemented gradient descent with momentum and found that both the test and train data accuracy increased. Lastly, we wanted to get the most out of our limited data so we attempted to implemented mini batches but faced a division by zero error and chose not to pursue further because the mini batch sizes were not large and was a waste of vectorization.
# 
# When we use L2 Regularization and Gradient Descent with Momentum at the same time we don't get as accurate results as applying them separately. After these experiments we've gained a greater understanding on what will help to improve our model. Knowing that we are dealing with high bias, in the future it would help to add more layers or nodes to our neural network. Also the data we're working on might benefit from changing the activation functions. For our model we chose to stick with the activation functions and the model that from part 1 and optimize our data in different ways. 

# Here in this part of the project we have done some fine tuning of the parameters in order to get better results
# In order to do so we created a function that creates random hyper-parameters. As we learnt in the class that the best approach to do hyper-parameter tuning is to do randomly instead of using the grid approach.
# We first select a general range where the random hyper-parameter such as learning rate is created then we magnify that particular range to find the best hyper-parameter.

# ### Hyperparameter tuning ###
# 
# To begin with, we decided to leave beta parameter for momentum improvement as 0.9 as there is no large range to test it out with.

# In[21]:


beta = 0.9


# #### 1) Increase number of layers
# 
# We experimented with several different layer types and decided to leave model with dimentions of 5 and layers . To begin with we knew that each new layer dimansion had to decrease and thus started with bigger values like 100 and slowly decreased this number with each layer.
# 
# After trying several versions for both original model and L2_momentum model we decided to move to the next hyperparameters with given layers:

# In[31]:


new_layers_dims = [11, 100, 30, 7,4,1]
parameters4 = L_layer_model(X_train, y_train, new_layers_dims, learning_rate = learning_rate, num_iterations = num_iterations, print_cost = print_cost, beta = beta, lambd = lambd)
print("Train Accuracy")
pred_train = predict_only_prob(X_train.T, y_train.T, parameters4)
print(pred_train)
print("Test Accuracy")
pred_test = predict_only_prob(X_test.T, y_test.T, parameters4)
print(pred_test)


# In[32]:


final_layers_dims = [11, 120, 50, 19, 7,4,1]

parameters1 = L_layer_model_L2_momentum(X_train, y_train, final_layers_dims, learning_rate = learning_rate, num_iterations = num_iterations, print_cost = print_cost, beta = beta, lambd = lambd)
print("Train Accuracy")
pred_train = predict2_only_prob(X_train.T, y_train.T, parameters1) 
print(pred_train)
print("Test Accuracy")
pred_test = predict2_only_prob(X_test.T, y_test.T, parameters1)
print(pred_test)


# #### 2) Increase number of iterations
# 
# In order to find the best number of iterations (or several other hyperparameters) we created a function that creates random hyper-parameters. As we learnt in the class that the best approach to do hyper-parameter tuning is to do randomly instead of using the grid approach.
# 
# We first select a general range where the random hyper-parameter such as learning rate is created then we magnify that particular range to find the best hyper-parameter.

# In[24]:


import random
random.seed(1)


# Next function is commented out as it can take some time for it to comlete its calculations. However, from the first range of [1000, 10000] of iterations the best number we got were in between 4000 and 6000 of iterations. For the better and more precise results we tried randomised number with range [4000, 6000]that gave the best result of 4800 and second best of 5600. We decided to settle on 4800

# In[25]:


# max_accuracy = 0
# best_num_iterations = 0
# for i in range(15):
#     rand_iteration = int(random.uniform(40, 60))*100
#     parameters1 = L_layer_model_L2_momentum(X_train, y_train, layers_dims, learning_rate = learning_rate, num_iterations = rand_iteration, print_cost = False, beta = beta, lambd = lambd)
#     pred_train = predict2_only_prob(X_train.T, y_train.T, parameters1) 
#     if(pred_train > max_accuracy):
#         max_accuracy = pred_train
#         best_num_iterations = rand_iteration
#     print(rand_iteration, pred_train)
# print(best_num_iterations, max_accuracy)


# After letting above function compute for several minutes we get best accuracy at around 79% as seen below.

# In[26]:


best_num_iterations = 4800 
#5600


# In[27]:


parameters1 = L_layer_model_L2_momentum(X_train, y_train, layers_dims, learning_rate = learning_rate, num_iterations = best_num_iterations, print_cost = print_cost, beta = beta, lambd = lambd)
print("Train Accuracy")
pred_train = predict2_only_prob(X_train.T, y_train.T, parameters1) 
print(pred_train)
print("Test Accuracy")
pred_test = predict2_only_prob(X_test.T, y_test.T, parameters1)
print(pred_test)


# #### 3) Better learning rate
# 
# For the learning rate we implemented the same randomised method as with number of iterations. After several tries and countless minutes later we were left with 2 best learning rates (best 0.2 and second best 0.5). To find the most effective learning rate we tried a randomised function in range [0.200, 0.300]. With the result of 0.265 we can get test accuracy close to 85% as seen below.

# In[28]:


# max_accuracy = 0
# best_alpha = 0
# for i in range(15):
#     rand_alpha = int(random.uniform(200, 300))/1000
#     parameters1 = L_layer_model_L2_momentum(X_train, y_train, layers_dims, learning_rate = rand_alpha, num_iterations = best_num_iterations, print_cost = False, beta = beta, lambd = lambd)
#     pred_train = predict2_only_prob(X_train.T, y_train.T, parameters1) 
#     if(pred_train > max_accuracy):
#         max_accuracy = pred_train
#         best_alpha = rand_alpha
#     print(rand_alpha, pred_train)
# print(best_alpha, max_accuracy)


# In[29]:


best_alpha = 0.265 
#0.5


# In[30]:


parameters1 = L_layer_model_L2_momentum(X_train, y_train, layers_dims, learning_rate = best_alpha, num_iterations = best_num_iterations, print_cost = print_cost, beta = beta, lambd = lambd)
print("Train Accuracy")
pred_train = predict2_only_prob(X_train.T, y_train.T, parameters1) 
print(pred_train)
print("Test Accuracy")
pred_test = predict2_only_prob(X_test.T, y_test.T, parameters1)
print(pred_test)


# **Grading**: 
# - Part 1 code for L_layer_model(): **10 marks**
# - Part 2:
#   - Dataset choice and analysis: **10 marks**
#   - Building your model:
#     - Basic model: **5 marks**
#     - Implementing improvements to model: **15 marks**
#   - Idea, Code, Experiment Cycle: **20 marks**
# 
# **Total** for project: **60 marks**

# **Submission**: Submit a zip file containing all of the files/folders for your project. Make sure all files are included; do not assume we have certain files already.
