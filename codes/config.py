#!/usr/bin/env python

import numpy as np, cv2, os, time, shutil, tensorflow as tf, tqdm, datetime, re
import random, math, json

#===============================================================================

classNames = [ 'dog', 'cat' ]
nClasses = len( classNames )    # Number of classes.

# Dictionary that refers to class index with key as class names.
className2labelIdx = { i: idx for idx, i in enumerate( classNames ) }

learningRate = 0.0001
batchSize = 100
nEpochs = 38

# The size estimate of the whole set of images is:
# maxHeight: 768, minHeight: 32, maxWidth: 1050, minWidth: 37
# avgHeight: 360.3, avgWidth: 404.14

inputImgH, inputImgW = 224, 224

# Some model design parameters.
modelName = 'model'
seedWB = 33     # Seed for initializing weight and bias variable initializers.
leak = 0.1      # Amount of leak of the leaky relu (in case we use it).
keepProb = 0.5      # Probability of dropout in fully connected layers.

logDirPath = './logs'       # Directory to save training logs.
ckptDirPath = './temp'      # Directory to save ckeckpoints.
savedCkptName = 'classifier'    # Name of saved checkpoints.
nSavedCkpt = 5      # Number of saved checkpoints.
printInterval = 1   # Intervals of printing training status on console.

modelSaveInterval = 1       # Interval at which model parameters are saved.

#===============================================================================

# Testing done. Test Accuracy: 93.120 %, Testing time: 2782.748 sec

#===============================================================================

if __name__ == '__main__':
    
    #class classifier( object ):
        #def __init__(self, y):
            #self.layerOut = {}
            #pass
        #def model(self, x):
            #x = x + y
            #self.layerOut[1] = x
            #x = x + y
            #self.layerOut[2] = x
            #x = x + y
            #self.layerOut[3] = x
            #x = x + y
            #self.layerOut[4] = x
            #x = x + y
            #self.layerOut[5] = x
            #return x
    
    #p = tf.Variable(1, name='x')
    #y = tf.Variable(2, name='y')
    #c = classifier(y)
    #o = c.model(p)
    ##y = x + 3
    ##z = y + 4
    
    #with tf.Session() as sess:
        #sess.run( tf.global_variables_initializer() )
        ##outX = sess.run(x)
        ##outY = sess.run(y)
        ##outZ = sess.run(z)
        #out, d = sess.run([o, c.layerOut])
    
    #print( out )
    #print( d )
    
    pass
