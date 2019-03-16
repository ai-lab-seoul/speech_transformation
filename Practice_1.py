import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 8)

import gc
import os
import ast
import sys
import configparser
import glob
import librosa
import librosa.display
from scipy.stats import mode

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras import backend as K


voice_path = 'D:\\Python\\Study\\Project\\speech_transformation\\data'
#print(os.listdir(voice_path))
#print(type(os.listdir(voice_path)))


def load_voice(voice_folder):
    voice_specs = []
    idx_to_voice = []
    person_to_idx = {}
    people = []

    for person in os.listdir(voice_folder):
        person_to_idx[person] = len(person_to_idx)
        people.append(person)
        person_folder = os.path.join(voice_folder, person)

        for data in os.listdir(person_folder):
            if data.endswith('.wav'):
                signal, sr = librosa.load(os.path.join(person_folder, data), duration=30)
                melspec = librosa.feature.melspectrogram(signal, sr = sr).T
                voice_specs.append(melspec)
                idx_to_voice.append(person_to_idx[person])

    return voice_specs, people, person_to_idx, idx_to_voice

voice_specs, people, person_to_idx, idx_to_voice = load_voice(voice_path)
print(voice_specs)
print(people)
print(person_to_idx)
print(idx_to_voice)
voice_specs[0].shape


librosa.display.specshow(librosa.power_to_db(voice_specs[1].T,
                                              ref=np.max),
                          y_axis='mel', fmax=8000,
                          x_axis='time')

def show_spectogram(show_person):
    show_person = person_to_idx[show_person]
    specs = []
    for spec, person in zip(voice_specs, idx_to_voice):
        if show_person == person:
            specs.append(spec)
            if len(specs) == 25:
                break
    if not specs:
        return 'not found!'
    x = np.concatenate(specs, axis=1)
    x = (x - x.min()) / (x.max() - x.min())
    plt.imshow((x *20).clip(0, 1.0))

show_spectogram('김신영')

#준혁

def cnn_model(input_shape):


    inputs = Input(input_shape)
    """
    Define the input 
    Unlike the Sequential model, you must create and define 
    a standalone "Input" layer that specifies the shape of input 
    data. The input layer takes a "shape" argument, which is a 
    tuple that indicates the dimensionality of the input data.
    When input data is one-dimensional, such as the MLP, the shape 
    must explicitly leave room for the shape of the mini-batch size 
    used when splitting the data when training the network. Hence, 
    the shape tuple is always defined with a hanging last dimension.
    For instance, "(2,)", as in the example below:
    """

    x = inputs
    levels = 64

    for level in range(3):
        x = Conv1D(levels, 3, activation='relu')(x)
        '''
        keras.layers.Conv1D(filters, 
                            kernel_size,
                            strides=1,
                            padding='valid',
                            data_format='channels_last',
                            dilation_rate=1,
                            activation=None,
                            use_bias=True,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros',
                            kernel_regularizer=None,
                            bias_regularizer=None,
                            activity_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=None)
                            
        This layer creates a convolution kernel that is convolved with the layer input over a single spatial (or temporal) dimension to produce a tensor of outputs.
        
        Arguments
        ---------
        filters : int
            the dimensionality of the output space (i.e. the number of output filters in the convolution)
        kernel_size : int or tuple
            list of a single integer, specifying the length of the 1D 
        activation : 
            Activation function to use (tanh, sigmoid, relu...) If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
            
        Input shape
        -----------
        3D tensor with shape : (batch, steps, channels)
        
        Output shape
        ------------
        3D tensor with shape : (batch, new_steps, filters) steps value might have changed due to padding or strides.


        '''

        x = BatchNormalization()(x)
        '''
        keras.layers.BatchNormalization(axis=-1, 
                                        momentum=0.99, 
                                        epsilon=0.001, 
                                        center=True, scale=True, 
                                        beta_initializer='zeros', 
                                        gamma_initializer='ones', 
                                        moving_mean_initializer='zeros', 
                                        moving_variance_initializer='ones', 
                                        beta_regularizer=None, 
                                        gamma_regularizer=None, 
                                        beta_constraint=None, 
                                        gamma_constraint=None)
                                        
        Normalize the activations of the previous layer at each batch. 
            (i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1)
            
        Input shape
        -----------
        Arbitrary. Use the keyword argument input_shape (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.
        
        Output shape
        ------------
        Same shape as input.    

        '''

        x = MaxPooling1D(pool_size=2, strides=2)(x)
        '''
        keras.layers.MaxPooling1D(pool_size=2, 
                                  strides=None, 
                                  padding='valid', 
                                  data_format='channels_last')
                                  
        Max pooling operation for temporal data.
        
        Arguments
        ---------
        pool_size : int
            size of the max pooling windows
        strides : int or None
            Factor by which to downscale. E.g. 2 will halve the input. If None, it will default to pool_size.
        
        Input shape
        -----------
        If data_format ='channels_last': 
            3D tensor with shape:  (batch_size, steps, features)
        If data_format ='channels_first': 
            3D tensor with shape:  (batch_size, features, steps)
            
        Output shape
        ------------
        If data_format ='channels_last': 
            3D tensor with shape:  (batch_size, downsampled_steps, features)
        If data_format ='channels_first': 
            3D tensor with shape:  (batch_size, features, downsampled_steps)
        '''
        levels *= 2 # 64 -> 128 -> 256

    # Global Layers
    x = GlobalMaxPooling1D()(x)
    '''
    keras.layers.GlobalMaxPooling1D(data_format='channels_last')
    
    Global max pooling operation for temporal data.

    Input shape
    -----------
    If data_format ='channels_last': 
        3D tensor with shape:  (batch_size, steps, features)
    If data_format ='channels_first': 
        3D tensor with shape:  (batch_size, features, steps)
            
    Output shape
    ------------
    2D tensor with shape: (batch_size, features)        

    '''

    for fc in range(2):
        x = Dense(256, activation='relu')(x)
        '''
        keras.layers.Dense(units, 
                           activation=None, 
                           use_bias=True, 
                           kernel_initializer='glorot_uniform', 
                           bias_initializer='zeros', 
                           kernel_regularizer=None, 
                           bias_regularizer=None, 
                           activity_regularizer=None, 
                           kernel_constraint=None, 
                           bias_constraint=None)
                           
        Just your regular densely-connected NN layer.

        Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, 
        kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
        
        Arguments
        ---------
        units : Positive int 
            dimensionality of the output space.
        activation : 
            Activation function to use.
        
        Input shape
        -----------
        nD tensor with shape: (batch_size, ..., input_dim). The most common situation would be a 2D input with shape (batch_size, input_dim).
    
        Output shape
        ------------
        nD tensor with shape: (batch_size, ..., units). For instance, for a 2D input with shape (batch_size, input_dim), the output would have shape (batch_size, units).
        '''
        x = Dropout(0.5)(x)
        '''
        keras.layers.Dropout(rate, 
                             noise_shape=None, 
                             seed=None)
                             
        Applies Dropout to the input.
        Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
        
        Arguments
        ---------
        rate : float between 0 and 1. 
            Fraction of the input units to drop.
        
        '''

    labels = Dense(10, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[labels])
    '''
     Create the model
    After creating all of your model layers and connecting them 
    together, you must then define the model.
    As with the Sequential API, the model is the thing that you can
    summarize, fit, evaluate, and use to make predictions.
    Keras provides a "Model" class that you can use to create a model 
    from your created layers. It requires that you only specify the 
    input and output layers.
    
    This model will include all layers required in the computation
    '''
    sgd = keras.optimizers.SGD(lr=0.0003, momentum=0.9, decay=1e-5, nesterov=True)
    '''
    keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    
    Stochastic gradient descent optimizer.
    Includes support for momentum, learning rate decay, and Nesterov momentum.
    
    Arguments
    ---------
    lr: float >= 0. 
        Learning rate.
    momentum: float >= 0. 
        Parameter that accelerates SGD in the relevant direction and dampens oscillations.
    decay: float >= 0. 
        Learning rate decay over each update.
    nesterov: boolean. 
        Whether to apply Nesterov momentum.

    '''

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    '''
    compile(optimizer, 
            loss=None, 
            metrics=None, 
            loss_weights=None, 
            sample_weight_mode=None, 
            weighted_metrics=None, 
            target_tensors=None)
            
    Configures the model for training.
    
    Arguments
    ---------
    optimizer : String (name of optimizer) or optimizer instance. 
    
    loss : String (name of objective function) or objective function. 
        If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. 
        The loss value that will be minimized by the model will then be the sum of all individual losses.
    metrics : List of metrics to be evaluated by the model during training and testing. Typically you will use metrics=['accuracy']. 
        To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a': 'accuracy'}.
     
    '''
    return model

'''


'''


model = cnn_model((128, 128))
model.summary()

