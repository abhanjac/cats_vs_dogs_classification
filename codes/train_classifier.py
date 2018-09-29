#!/usr/bin/env python

from config import *
from utils import *

#===============================================================================

class CDclassifier( object ):
    '''
    This is a class for classifying the images as dogs or cats.
    '''
    def __init__( self ):
        '''
        Initializing the model.
        '''
        # Defining methods to initilize the model weights and biases.
        self.initW = tf.glorot_normal_initializer( seed=seedWB, dtype=tf.float32 )
        self.initB = tf.zeros_initializer()
        
        # Flag to indicate if the model is used for training or validation.
        # Based on this flag, the batch_normalization and dropout layers will 
        # be configured.
        self.isTraining = False
        
        # Dictionary to hold the individual outputs of each model layer.
        self.layerOut = {}
        
        # Defining the optimizer class.
        # This is done here because we will be needing the optimizerName in the 
        # test function as well. If we will define the optimizer in the train
        # function, then during testing when the train function is not called,
        # the optimizerName will not be initialized. So it is defined in init
        # such that it gets defined as the class object is initialized.
        self.optimizer = tf.train.AdamOptimizer( learning_rate=learningRate )
        self.optimizerName = self.optimizer.get_name()   # Name of optimizer.   

#===============================================================================
    
    def model( self, x ):
        '''
        Defines the model structure.
        '''
        x = tf.convert_to_tensor(x)

        layerIdx = '1'      # Input size 224 x 224 x 3 (H x W x C).
        layerName = 'conv' + layerIdx
        x = tf.layers.conv2d( x, kernel_size=(3,3), filters=32, padding='SAME', \
                use_bias=False, activation=None, name=layerName, \
                kernel_initializer=self.initW, bias_initializer=self.initB )
        self.layerOut[ layerName ] = x
        
        layerName = 'activation' + layerIdx
        x = tf.nn.relu(x)
        self.layerOut[ layerName ] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tf.layers.batch_normalization( x, training=self.isTraining, name=layerName )
        self.layerOut[ layerName ] = x
        
        layerName = 'pooling' + layerIdx
        x = tf.layers.max_pooling2d( x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName )
        self.layerOut[ layerName ] = x
        
        # Output size 112 x 112 x 3 (H x W x C).

#-------------------------------------------------------------------------------

        layerIdx = '2'      # Input size 112 x 112 x 32 (H x W x C).
        layerName = 'conv' + layerIdx
        x = tf.layers.conv2d( x, kernel_size=(3,3), filters=64, padding='SAME', \
                use_bias=False, activation=None, name=layerName, \
                kernel_initializer=self.initW, bias_initializer=self.initB )
        self.layerOut[ layerName ] = x
        
        layerName = 'activation' + layerIdx
        x = tf.nn.relu(x)
        self.layerOut[ layerName ] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tf.layers.batch_normalization( x, training=self.isTraining, name=layerName )
        self.layerOut[ layerName ] = x
        
        layerName = 'pooling' + layerIdx
        x = tf.layers.max_pooling2d( x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName )
        self.layerOut[ layerName ] = x

        # Output size 56 x 56 x 64 (H x W x C).

#-------------------------------------------------------------------------------

        layerIdx = '3'      # Input size 56 x 56 x 64 (H x W x C).
        layerName = 'conv' + layerIdx
        x = tf.layers.conv2d( x, kernel_size=(3,3), filters=128, padding='SAME', \
                use_bias=False, activation=None, name=layerName, \
                kernel_initializer=self.initW, bias_initializer=self.initB )
        self.layerOut[ layerName ] = x
        
        layerName = 'activation' + layerIdx
        x = tf.nn.relu(x)
        self.layerOut[ layerName ] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tf.layers.batch_normalization( x, training=self.isTraining, name=layerName )
        self.layerOut[ layerName ] = x
        
        layerName = 'pooling' + layerIdx
        x = tf.layers.max_pooling2d( x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName )
        self.layerOut[ layerName ] = x

        # Output size 28 x 28 x 128 (H x W x C).

#-------------------------------------------------------------------------------

        layerIdx = '4'      # Input size 28 x 28 x 128 (H x W x C).
        layerName = 'conv' + layerIdx
        x = tf.layers.conv2d( x, kernel_size=(3,3), filters=256, padding='SAME', \
                use_bias=False, activation=None, name=layerName, \
                kernel_initializer=self.initW, bias_initializer=self.initB )
        self.layerOut[ layerName ] = x
        
        layerName = 'activation' + layerIdx
        x = tf.nn.relu(x)
        self.layerOut[ layerName ] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tf.layers.batch_normalization( x, training=self.isTraining, name=layerName )
        self.layerOut[ layerName ] = x
        
        layerName = 'pooling' + layerIdx
        x = tf.layers.max_pooling2d( x, pool_size=(2,2), strides=2, \
                                        padding='SAME', name=layerName )
        self.layerOut[ layerName ] = x

        # Output size 14 x 14 x 256 (H x W x C).

#-------------------------------------------------------------------------------

        layerIdx = '5'      # Input size 14 x 14 x 256 (H x W x C).
        layerName = 'conv' + layerIdx
        x = tf.layers.conv2d( x, kernel_size=(3,3), filters=512, padding='SAME', \
                use_bias=False, activation=None, name=layerName, \
                kernel_initializer=self.initW, bias_initializer=self.initB )
        self.layerOut[ layerName ] = x
        
        layerName = 'activation' + layerIdx
        x = tf.nn.relu(x)
        self.layerOut[ layerName ] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tf.layers.batch_normalization( x, training=self.isTraining, name=layerName )
        self.layerOut[ layerName ] = x
        
        #layerName = 'pooling' + layerIdx
        #x = tf.layers.max_pooling2d( x, pool_size=(2,2), strides=2, \
                                        #padding='SAME', name=layerName )
        #self.layerOut[ layerName ] = x

        # Output size 14 x 14 x 512 (H x W x C).

#-------------------------------------------------------------------------------

        layerIdx = '6'      # Input size 14 x 14 x 512 (H x W x C).
        layerName = 'conv' + layerIdx
        x = tf.layers.conv2d( x, kernel_size=(1,1), filters=256, padding='SAME', \
                use_bias=False, activation=None, name=layerName, \
                kernel_initializer=self.initW, bias_initializer=self.initB )
        self.layerOut[ layerName ] = x
        
        layerName = 'activation' + layerIdx
        x = tf.nn.relu(x)
        self.layerOut[ layerName ] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tf.layers.batch_normalization( x, training=self.isTraining, name=layerName )
        self.layerOut[ layerName ] = x
        
        #layerName = 'pooling' + layerIdx
        #x = tf.layers.max_pooling2d( x, pool_size=(2,2), strides=2, \
                                        #padding='SAME', name=layerName )
        #self.layerOut[ layerName ] = x

#------------------------------------------------------------------------------

        layerIdx = '7'      # Input size 14 x 14 x 256 (H x W x C).
        layerName = 'conv' + layerIdx
        x = tf.layers.conv2d( x, kernel_size=(3,3), filters=512, padding='SAME', \
                use_bias=False, activation=None, name=layerName, \
                kernel_initializer=self.initW, bias_initializer=self.initB )
        self.layerOut[ layerName ] = x
        
        layerName = 'activation' + layerIdx
        x = tf.nn.relu(x)
        self.layerOut[ layerName ] = x
        
        layerName = 'batchNorm' + layerIdx
        x = tf.layers.batch_normalization( x, training=self.isTraining, name=layerName )
        self.layerOut[ layerName ] = x
        
        # This next layer will behave as the global average pooling layer. So 
        # the padding has to be 'VALID' here to give a 1x1 output size. 
        layerName = 'pooling' + layerIdx
        x = tf.layers.max_pooling2d( x, pool_size=(finalLayerH, finalLayerW), strides=1, \
                                        padding='VALID', name=layerName )
        
        x = tf.layers.flatten(x)    # This will keep the 0th dimension 
        # (batch size dimension) the same and flatten the rest of the elements 
        # (which has shape 1x1x64 right now) into a single dimension (of size 64).
        
        self.layerOut[ layerName ] = x

#-------------------------------------------------------------------------------

        layerIdx = '8'      # Input size 1 x 1 x 512 (H x W x C).
        layerName = 'dense' + layerIdx
        # No activation applied to the fc layer. Can be added outside if needed.
        x = tf.layers.dense( x, units=nClasses, use_bias=True, name=layerName, \
                    kernel_initializer=self.initW, bias_initializer=self.initB )
        self.layerOut[ layerName ] = x
        
        layerName = 'dropout' + layerIdx
        x = tf.layers.dropout( x, rate=keepProb, training=self.isTraining, name=layerName )
        self.layerOut[ layerName ] = x

        return x
    
#===============================================================================
    
    def loss( self, logits, labels ):
        '''
        Defines the loss function.
        The sigmoid_cross_entropy_with_logits loss is chosen as it measures 
        probability error in classification tasks where each class is 
        independent and not mutually exclusive. E.g. one could perform 
        multilabel classification where a picture can contain both cat and dog.
        '''
        labels = tf.convert_to_tensor( labels )
        labels = tf.cast( labels, tf.int32 )    # Converting labels to int 
        # (as tf.one_hot does not accept float values).
        
        oneHotLabels = tf.one_hot( labels, depth=nClasses, dtype=tf.float32 )
        # The one hot vectors are created in the place of the original values 
        # as a list. Hence a tensor of shape [100, 13, 13, 5, 1] becomes a new
        # tensor of shape [100, 13, 13, 5, 1, 80] (if there were 80 classes). 
        # But the shape needed here is [100, 13, 13, 5, 80]. Hence merging the
        # last two dimensions.
        self.oneHotLabels = tf.reshape( oneHotLabels, shape=( -1, nClasses ) )
        
        # This returns a tensor of same shape as logits with the componentwise 
        # logistic losses.
        #lossTensor = tf.nn.sigmoid_cross_entropy_with_logits( logits=logits, \
                                        #labels=self.oneHotLabels )
        lossTensor = tf.nn.softmax_cross_entropy_with_logits_v2( logits=logits, \
                                        labels=self.oneHotLabels )
        
        # Return the average loss over this batch.
        #return tf.reduce_sum( lossTensor )
        return tf.reduce_mean( lossTensor )
    
#==============================================================================

    def train( self, trainDir=None, validDir=None ):
        '''
        Trains the model.
        '''
        if trainDir == None or validDir == None:
            print( '\ntrainDir or validDir not provided. Aborting...' )
            return
                
        # SET INPUTS AND LABELS.
        # Batch size will be set during runtime as the last batch may not be of
        # the same size of the other batches.
        x = tf.placeholder( dtype=tf.float32, name='xPlaceholder', \
                            shape=[ None, inputImgH, inputImgW, 3 ] )
        y = tf.placeholder( dtype=tf.int32, name='yPlaceholder', \
                            shape=[ None, 1 ])      # Labels are int.
        
        # EVALUATE MODEL OUTPUT.
        with tf.variable_scope( modelName, reuse=tf.AUTO_REUSE ):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction.
            # List of model variables.
            listOfModelVars = []
            for v in tf.global_variables():
                listOfModelVars.append( v )
                #print( 'Model variable: ', v )        
        
        # CALCULATE LOSS.
        loss = self.loss( logits=predLogits, labels=y )
        
        # DEFINE OPTIMIZER AND PERFORM BACKPROP.
        optimizer = self.optimizer
        
        # While executing an operation (such as trainStep), only the subgraph 
        # components relevant to trainStep will be executed. The 
        # update_moving_averages operation (for the batch normalization layers) 
        # is not a parent of trainStep in the computational graph, so it will 
        # never update the moving averages by default. To get around this, 
        # we have to explicitly tell the graph in the following manner.
        update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
        with tf.control_dependencies( update_ops ):
            trainStep = optimizer.minimize( loss )

        # List of model and optimizer variables.
        listOfModelAndOptimizerVars = []
        for v in tf.global_variables():
            listOfModelAndOptimizerVars.append( v )
            #print( 'Model or Optimizer variable: ', v )            
            
        # CREATING LOG FILE.
        # Create a folder to hold the training log files if not already there.
        if not os.path.exists( logDirPath ):
            os.makedirs( logDirPath )
        
        # Create training log file.
        logFileName = 'training_log{}.txt'.format( timeStamp() )
        logFilePath = os.path.join( logDirPath, logFileName )
        
        # CREATE A LISTS TO HOLD ACCURACY AND LOSS VALUES.
        # This list will have strings for each epoch. Each of these strings 
        # will be like the following:
        # "epoch, learningRate, trainLoss, trainAcc, validAcc"
        statistics = []
        # Format of the statistics values.
        statisticsFormat = 'epoch, learningRate, trainLoss, trainAcc, validAcc'
                        
#-------------------------------------------------------------------------------
        
        # START SESSION.
        with tf.Session() as sess:
            print( '\nStarting session. Optimizer: {}, Learning Rate: {}, ' \
                'Batch Size: {}'.format( self.optimizerName, learningRate, batchSize ) )
            
            self.isTraining = True    # Enabling the training flag. 
            
            # Define model saver.
            # Finding latest checkpoint and the latest completed epoch.
            metaFilePath, ckptPath, latestEpoch = findLatestCkpt( ckptDirPath, \
                                                    training=self.isTraining )
            startEpoch = latestEpoch + 1    # Start from the next epoch.
            
            if ckptPath != None:    # Only when some checkpoint is found.
                with open( ckptPath + '.json', 'r' ) as infoFile:
                    info = json.load( infoFile )
                    
                if info[ 'learningRate' ] == learningRate and \
                   info[ 'batchSize' ] == batchSize:
                    # If these two are same, only then we can load the older 
                    # model checkpoint with all the variables.
                    # So define the saver with the list of all the variables.
                    saver = tf.train.Saver( listOfModelAndOptimizerVars )
                    saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.
                    saver.restore( sess, ckptPath )
                    
                    # Since all variables will be loaded here, so we do not need
                    # to initialize any other variables.

                    status = '\nReloaded ALL variables from checkpoint: {}'.format( \
                                                                ckptPath )
                    # Update training log.
                    with open( logFilePath, 'a', buffering=1 ) as trainingLogs:
                       trainingLogs.write( status )
                    print( status )
                
                else:
                    # else load only weight and biases and skip optimizer 
                    # variables. Otherwise there will be errors.
                    # So define the saver with the list of only model variables.
                    saver = tf.train.Saver( listOfModelVars )
                    saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.
                    saver.restore( sess, ckptPath )
                    # Since only the model variables will be loaded here, so we
                    # have to initialize other variables (optimizer variables)
                    # separately.
                    otherVariables = [ v for v in listOfModelAndOptimizerVars \
                                           if v not in listOfModelVars ]
                    sess.run( tf.variables_initializer( otherVariables ) )

                    # The previous saver will only save the listOfModelVars.
                    # as it is defined using only those variables (as the 
                    # checkpoint can only give us those values as valid for
                    # restoration). But now since we have all the varibles 
                    # loaded and all the new ones initialized, so we redifine 
                    # the saver to include all the variables. So that while 
                    # saving in the end of the epoch, it can save all the 
                    # listOfModelAndOptimizerVars (and not just the 
                    # listOfModelVars).
                    saver = tf.train.Saver( listOfModelAndOptimizerVars )
                    saver.max_to_keep = nSavedCkpt    # Save upto 5 checkpoints.
                    # Load mean and std values.
                    mean, std = info[ 'mean' ], info[ 'std' ]
                    
                    status = '\nCurrent parameters:\nlearningRate: {}, batchSize: {}' \
                       '\nPrevious parameters (inside checkpoint {}):\nlearningRate:' \
                       '{}, batchSize: {}\nThey are different.\nSo reloaded only ' \
                       'MODEL variables from checkpoint: {}\nAnd initialized ' \
                       'other variables.'.format( learningRate, batchSize, ckptPath, \
                       info[ 'learningRate' ], info[ 'batchSize' ], ckptPath )

                    # Update training log.
                    with open( logFilePath, 'a', buffering=1 ) as trainingLogs:
                       trainingLogs.write( status )
                    print( status )
                    
                # Reloading accuracy and loss statistics, mean and std from checkpoint.
                statistics = info[ 'statistics' ]
                mean = np.array( info[ 'mean' ] )
                std = np.array( info[ 'std' ] )
                maxValidAcc = info[ 'maxValidAcc' ]

            else:
                # When there are no valid checkpoints initialize the saver to 
                # save all parameters.
                saver = tf.train.Saver( listOfModelAndOptimizerVars, \
                        max_to_keep=nSavedCkpt )   # Save upto 5 checkpoints.
                sess.run( tf.global_variables_initializer() )
               
                # Calculate mean and std.
                mean, std = datasetMeanStd( trainDir )
                maxValidAcc = 0

#-------------------------------------------------------------------------------
                    
            print( '\nStarted Training...' )
            
            for epoch in range( startEpoch, nEpochs+1 ):
                # epoch will be numbered from 1 to 150 if there are 150 epochs.
                # Is is not numbered from 0 to 149.

                self.isTraining = True    # Enabling training flag at the start of 
                # the training phase of each epoch as it will be disabled in the 
                # corresponding validaton phase. 

                epochProcessTime = time.time()
                
                # TRAINING PHASE.
                # This list contains the filepaths for all the images in trainDir.
                listOfRemainingTrainImg = [ os.path.join( trainDir, i ) for i \
                                                in os.listdir( trainDir ) ]
                
                nTrainBatches = math.ceil( len( listOfRemainingTrainImg ) / batchSize )
                trainLoss, trainAcc = 0.0, 0.0
                
                trainBatchIdx = 0    # Counts the number of batches processed.
                
                # Storing information of current epoch for recording later in statistics.
                # This is recorded as a string for better visibility in the json file.
                currentStatistics = '{}, {}, '.format( epoch, learningRate )
                
                # Scan entire training dataset.
                while len( listOfRemainingTrainImg ) > 0:

                    trainBatchProcessTime = time.time()

                    # Creating batches. Also listOfRemainingTrainImg is updated.
                    trainImgBatch, trainLabelBatch, listOfRemainingTrainImg = \
                        createBatch( listOfRemainingTrainImg, batchSize, \
                                     shuffle=True, mean=mean, std=std )

                    trainImgBatch = np.array( trainImgBatch )
                    trainLabelBatch = np.array( trainLabelBatch )
                    trainLabelBatch = np.expand_dims( trainLabelBatch, axis=1 )  
                    # 100x1 array now.
                    
                    feedDict = { x: trainImgBatch, y: trainLabelBatch }
                    trainLayerOut = sess.run( self.layerOut, feed_dict=feedDict )
                    trainPredLogits = sess.run( predLogits, feed_dict=feedDict )
                    trainBatchLoss = sess.run( loss, feed_dict=feedDict )
                    sess.run( trainStep, feed_dict=feedDict )
                    
                    trainLoss += ( trainBatchLoss / nTrainBatches )
                    
                    # The trainPredLogits is an array of logits. It needs to be 
                    # converted to sigmoid to get probability and then we need
                    # to extract the max index to get the labels.
                    trainPredProb = np.array( [ softmax( trainPredLogits[r] ) \
                                                for r in range( batchSize ) ] )
                    trainPredLabel = np.argmax( trainPredProb, axis=1 )
                    trainPredLabel = np.expand_dims( trainPredLabel, axis=1 )
                    
                    ## Printing current batch prediction and actual labels.
                    #for p in range( batchSize ):
                        #print( 'trainLabelBatch: {} ; trainPredLabel: {} ; ' \
                               #'trainPredProb: {}'.format( trainLabelBatch[p], \
                                   #trainPredLabel[p], trainPredProb[p] ) )
                    
                    matches = np.array( trainPredLabel == trainLabelBatch, dtype=int )
                    trainAcc += ( 100*np.sum( matches ) ) / ( nTrainBatches*batchSize )
                    
                    trainBatchIdx += 1
                    trainBatchProcessTime = time.time() - trainBatchProcessTime

                    # Printing and logging current status of training.
                    status1 = '\nEpoch: {}/{},\tBatch: {}/{},\tBatch loss: ' \
                        '{:0.6f},\tBatch time: {:0.3f} sec '.format( epoch, \
                        nEpochs, trainBatchIdx, nTrainBatches, trainBatchLoss, \
                        trainBatchProcessTime )
                    
                    # Update training log.
                    with open( logFilePath, 'a', buffering=1 ) as trainingLogs:
                        trainingLogs.write( status1 )
                    
                    # print the status on the terminal every 10 batches.
                    if trainBatchIdx % printInterval == 0:
                        print( status1, end='' )
                
                # Recording training loss and accuracy in current statistics string.
                currentStatistics += '{}, {}, '.format( trainLoss, trainAcc )
                
#-------------------------------------------------------------------------------
                
                # VALIDATION PHASE.
                # This list contains the filepaths for all the images in validDir.
                listOfRemainingValidImg = [ os.path.join( validDir, i ) for i \
                                                in os.listdir( validDir ) ]
                
                nValidBatches = math.ceil( len( listOfRemainingValidImg ) / batchSize )
                validAcc = 0.0
                self.isTraining = False    # Disabling the training flag.
                validBatchIdx = 0    # Counts the number of batches processed.

                status1 = '\n\nValidation phase for epoch {}.'.format( epoch )
                
                # Update training log.
                with open( logFilePath, 'a', buffering=1 ) as trainingLogs:
                                 trainingLogs.write( status1+'\n' )
            
                print( '\nValidation phase for epoch {}.'.format( epoch ) )
                
                # Scan entire validation dataset.
                while len( listOfRemainingValidImg ) > 0:
                    
                    validBatchProcessTime = time.time()

                    # Creating batches. Also listOfRemainingValidImg is updated.
                    # The shuffle is off for validation and the mean and std are
                    # the same as calculated on the training set.
                    validImgBatch, validLabelBatch, listOfRemainingValidImg = \
                        createBatch( listOfRemainingValidImg, batchSize, \
                                     shuffle=False, mean=mean, std=std)

                    validImgBatch = np.array( validImgBatch )
                    validLabelBatch = np.array( validLabelBatch )
                    validLabelBatch = np.expand_dims( validLabelBatch, axis=1 )  
                    # 100x1 array now.
                    
                    feedDict = { x: validImgBatch, y: validLabelBatch }
                    validLayerOut = sess.run( self.layerOut, feed_dict=feedDict )
                    validPredLogits = sess.run( predLogits, feed_dict=feedDict )
                                        
                    # The validPredLogits is an array of logits. It needs to be 
                    # converted to sigmoid to get probability and then we need
                    # to extract the max index to get the labels.
                    validPredProb = np.array( [ softmax( validPredLogits[r] ) \
                                                for r in range( batchSize ) ] )
                    validPredLabel = np.argmax( validPredProb, axis=1 )
                    validPredLabel = np.expand_dims( validPredLabel, axis=1 )
                    
                    matches = np.array( validPredLabel == validLabelBatch, dtype=int )
                    validAcc += ( 100*np.sum( matches ) ) / ( nValidBatches*batchSize )
                    
                    validBatchIdx += 1
                    validBatchProcessTime = time.time() - validBatchProcessTime

                    # Printing and logging current status of validation.
                    status1 = '\nEpoch: {}/{},\tBatch: {}/{},\tBatch time: ' \
                        '{:0.3f} sec '.format( epoch, nEpochs, validBatchIdx, \
                        nValidBatches, validBatchProcessTime )
                    
                    # Update training log.
                    with open( logFilePath, 'a', buffering=1 ) as trainingLogs:
                        trainingLogs.write( status1 )
                    
                    # print the status on the terminal every 10 batches.
                    if validBatchIdx % printInterval == 0:     
                        print( status1, end='' )
                        
                # Recording validation accuracy in current statistics string.
                currentStatistics += '{}'.format( validAcc )
                
                # Noting accuracy after the end of all batches.
                statistics.append( currentStatistics )
                
#-------------------------------------------------------------------------------
                
                # STATUS UPDATE.
                epochProcessTime = time.time() - epochProcessTime
                
                # Printing and logging current epoch.
                status3 = '\nEpoch {} done. Epoch time: {:0.3f} sec, Train ' \
                    'loss: {:0.6f}, Train Accuracy: {:0.3f} %, Valid Accuracy: {:0.3f} %' \
                    .format( epoch, epochProcessTime, trainLoss, trainAcc, validAcc )
                
                # Update training log.
                with open( logFilePath, 'a', buffering=1 ) as trainingLogs:
                    trainingLogs.write( '\n'+status3+'\n' )
                    
                print( '\n' + status3 )

                # Saving the variables at some intervals, only if there is 
                # improvement in validation accuracy.
                if epoch % modelSaveInterval == 0 and validAcc > maxValidAcc:
                    ckptSavePath = os.path.join( ckptDirPath, savedCkptName )
                    saver.save( sess, save_path=ckptSavePath, global_step=epoch )
                    
                    maxValidAcc = validAcc      # Updating the maxValidAcc.
                    
                    # Saving the important info like learning rate, batch size,
                    # and training error for the current epoch in a json file.
                    # Converting the mean and std into lists before storing as
                    # json cannot store numpy arrays. And also saving the training
                    # and validation loss and accuracy statistics.
                    jsonInfoFilePath = ckptSavePath + '-' + str( epoch ) + '.json'
                    with open( jsonInfoFilePath, 'w' ) as infoFile:
                        infoDict = { 'epoch': epoch, 'batchSize': batchSize, \
                                     'learningRate': learningRate, \
                                     'mean': list( mean ), 'std': list( std ), \
                                     'maxValidAcc': maxValidAcc, \
                                     'statisticsFormat': statisticsFormat, \
                                     'statistics': statistics }
                        
                        json.dump( infoDict, infoFile, sort_keys=False, \
                                   indent=4, separators=(',', ': ') )
                    
                    status2 = '\nCheckpoint saved.'
                    
                    # Update training log.
                    with open( logFilePath, 'a', buffering=1 ) as trainingLogs:
                        trainingLogs.write( '\n'+status2+'\n' )

                    print( status2 )
                    #print( status2, file=trainingLogs )
                
                # Updating the maxValidAcc value.
                elif validAcc > maxValidAcc:      maxValidAcc = validAcc
                
        
        self.isTraining = False   # Indicates the end of training.
        print( '\nTraining completed with {} epochs.'.format( nEpochs ) )

#===============================================================================

    def test( self, testDir=None ):
        '''
        Tests the model.
        '''
        if testDir == None:
            print( '\ntestDir not provided. Aborting...' )
            return
        
        self.isTraining = False    # Disabling the training flag.
        
        # SET INPUTS AND LABELS.
        # Batch size will be set during runtime as the last batch may not be of
        # the same size of the other batches.
        x = tf.placeholder( dtype=tf.float32, name='xPlaceholder', \
                            shape=[ None, inputImgH, inputImgW, 3 ] )
        y = tf.placeholder( dtype=tf.int32, name='yPlaceholder', \
                            shape=[ None, 1 ])      # Labels are int.
        
        # EVALUATE MODEL OUTPUT.
        with tf.variable_scope( modelName, reuse=tf.AUTO_REUSE ):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction.
            # List of model variables.
            listOfModelVars = []
            for v in tf.global_variables():
                listOfModelVars.append( v )
                #print( 'Model variable: ', v )        

#-------------------------------------------------------------------------------
        
        # START SESSION.
        with tf.Session() as sess:
            # Define model saver.
            # Finding latest checkpoint and the latest completed epoch.
            metaFilePath, ckptPath, _ = findLatestCkpt( ckptDirPath, \
                                            training=self.isTraining )
            
            if ckptPath != None:    # Only when some checkpoint is found.
                saver = tf.train.Saver( listOfModelVars )
                saver.restore( sess, ckptPath )
                
                status = '\nReloaded ALL variables from checkpoint: {}'.format( \
                                                            ckptPath )
            else:
                # When there are no valid checkpoints.
                print( '\nNo valid checkpoints found. Aborting...' )
                return

            with open( ckptPath + '.json', 'r' ) as infoFile:
                info = json.load( infoFile )
                
            # Reloading mean and std from checkpoint.
            mean = np.array( info[ 'mean' ] )
            std = np.array( info[ 'std' ] )

#-------------------------------------------------------------------------------
                    
            print( '\nStarted Testing...' )
            
            # TESTING PHASE.
            testingTime = time.time()
            
            # This list contains the filepaths for all the images in testDir.
            listOfRemainingTestImg = [ os.path.join( testDir, i ) for i \
                                            in os.listdir( testDir ) ]
            
            nTestBatches = math.ceil( len( listOfRemainingTestImg ) / batchSize )
            testAcc = 0.0
            
            testBatchIdx = 0    # Counts the number of batches processed.
            
            # Scan entire testing dataset.
            while len( listOfRemainingTestImg ) > 0:
                
                testBatchProcessTime = time.time()

                # Creating batches. Also listOfRemainingTestImg is updated.
                # The shuffle is off for validation and the mean and std are
                # the same as calculated on the training set.
                testImgBatch, testLabelBatch, listOfRemainingTestImg = \
                    createBatch( listOfRemainingTestImg, batchSize, \
                                 shuffle=False, mean=mean, std=std )

                testImgBatch = np.array( testImgBatch )
                testLabelBatch = np.array( testLabelBatch )
                testLabelBatch = np.expand_dims( testLabelBatch, axis=1 )  
                # 100x1 array now.
                
                feedDict = { x: testImgBatch, y: testLabelBatch }
                testLayerOut = sess.run( self.layerOut, feed_dict=feedDict )
                testPredLogits = sess.run( predLogits, feed_dict=feedDict )
                
                # The testPredLogits is an array of logits. It needs to be 
                # converted to sigmoid to get probability and then we need
                # to extract the max index to get the labels.
                testPredProb = np.array( [ softmax( testPredLogits[r] ) \
                                            for r in range( batchSize ) ] )
                testPredLabel = np.argmax( testPredProb, axis=1 )
                testPredLabel = np.expand_dims( testPredLabel, axis=1 )
                
                #for i in range(batchSize):
                    #print( testPredLabel[i], testLabelBatch[i] )
                
                matches = np.array( testPredLabel == testLabelBatch, dtype=int )
                testAcc += ( 100*np.sum( matches ) ) / ( nTestBatches*batchSize )
                
                testBatchIdx += 1
                testBatchProcessTime = time.time() - testBatchProcessTime

                # Printing current status of testing.
                # print the status on the terminal every 10 batches.
                if testBatchIdx % printInterval == 0:
                    print( '\nBatch: {}/{},\tBatch time: {:0.3f} sec '.format( \
                        testBatchIdx, nTestBatches, testBatchProcessTime ), end='' )

#-------------------------------------------------------------------------------

        testingTime = time.time() - testingTime
        print( '\n\nTesting done. Test Accuracy: {:0.3f} %, Testing time: ' \
            '{:0.3f} sec'.format( testAcc, testingTime  ) )
        
        return testAcc, testingTime

#===============================================================================

    def getWeight( self, varName=None ):
        '''
        This function takes in a weight variable name and gives its value as 
        output. These are the model variables.
        '''
        self.isTraining = False    # Disabling the training flag.

        #print(varName)
        desiredVar = None
                
        with tf.variable_scope( '', reuse=tf.AUTO_REUSE ):
            # Since this is not defined after any model or optimizer, so by 
            # default tf.global_variables() will give a list of all variables, 
            # both of the model and optimizer.
            for v in tf.global_variables():
                #print( v.name )
                if v.name == varName + ':0':
                    desiredVar = tf.get_variable( varName )
                    
#-------------------------------------------------------------------------------

        # Now we have to evaluate the value of the variable in a session.
        # START SESSION.
        with tf.Session() as sess:
            # Define model saver.
            # Finding latest checkpoint and the latest completed epoch.
            metaFilePath, ckptPath, _ = findLatestCkpt( ckptDirPath, \
                                                    training=self.isTraining )
            
            if ckptPath != None:    # Only when some checkpoint is found.
                saver = tf.train.Saver()
                saver.restore( sess, ckptPath )
            else:
                # When there are no valid checkpoints.
                print( '\nNo valid checkpoints found. Aborting...' )
                return

#-------------------------------------------------------------------------------

            # Evaluate the variable value.
            if desiredVar != None:    desiredVarValue = sess.run( desiredVar )
            else:    desiredVarValue = None
        
        return desiredVarValue

#===============================================================================

    def inference( self, imgRaw ):
        '''
        This function evaluates the output of the model on an unknown single 
        images only, i.e. can be used as the forward function as well. 
        It returns predicted labels and also the feature maps.
        '''
        self.isTraining = False
        
        h, w = imgRaw.shape[0], imgRaw.shape[1]
        
        if h > inputImgH or w > inputImgW:  intpol = cv2.INTER_LINEAR
        else:   intpol = cv2.INTER_AREA
        
        imgResized = cv2.resize( imgRaw, ( inputImgW, inputImgH ), \
                                        interpolation=intpol )
        s = list( imgResized.shape )
        if len(s) == 2:     # Single gray image.
            h, w = s[0], s[1]
            imgBatch = np.reshape( imgResized, ( 1, h, w, 1 ) )
        elif len(s) == 3 and imgResized.shape[2] == 3:    # Single color image.
            h, w, c = s[0], s[1], s[2]
            imgBatch = np.reshape( imgResized, ( 1, h, w, c ) )
            
        # At this point the input img array is a batch containing one 3 
        # channel image.

#-------------------------------------------------------------------------------

        x = tf.placeholder( dtype=tf.float32, name='xPlaceholder', \
                            shape=[ None, inputImgH, inputImgW, 3 ] )
                
        # EVALUATE MODEL OUTPUT.        
        with tf.variable_scope( modelName, reuse=tf.AUTO_REUSE ):
            # AUTO_REUSE flag is used so that no error is there when the same 
            # model parameters are used to check multiple images in sequence.
            predLogits = self.model(x)     # Model prediction.
            # List of model variables.
            listOfModelVars = []
            for v in tf.global_variables():
                listOfModelVars.append( v )
                #print( 'Model variable: ', v )

#-------------------------------------------------------------------------------

        # START SESSION.
        mean, std = 0.0, 1.0

        with tf.Session() as sess:
            # Define model saver.
            # Finding latest checkpoint and the latest completed epoch.
            metaFilePath, ckptPath, _ = findLatestCkpt( ckptDirPath, \
                                                training=self.isTraining )
            
            if ckptPath != None:    # Only when some checkpoint is found.
                saver = tf.train.Saver( listOfModelVars )
                saver.restore( sess, ckptPath )
                
                print( '\nReloaded ALL variables from checkpoint: {}'.format( \
                                                            ckptPath ) )
                with open( ckptPath + '.json', 'r' ) as infoFile:
                    info = json.load( infoFile )
                    
                    # Mean and std of training set obtained from checkpoint.
                    mean = np.array( info[ 'mean' ] )
                    std = np.array( info[ 'std' ] )
        
            else:
                # When there are no valid checkpoints.
                print( '\nNo valid checkpoints found. Aborting...' )
                return

#-------------------------------------------------------------------------------

            # Normalizing by mean and std as done in case of training.
            imgBatch = (imgBatch - mean) / std

            feedDict = { x: imgBatch }
            inferLayerOut = sess.run( self.layerOut, feed_dict=feedDict )
            inferPredLogits = sess.run( predLogits, feed_dict=feedDict )

        # The inferPredLogits is an array of logits. It needs to be 
        # converted to sigmoid to get probability and then we need
        # to extract the max index to get the labels.
        inferPredProb = np.array( softmax( inferPredLogits[0] ) )
        inferPredLabel = np.argmax( inferPredProb )
        
        return inferLayerOut, inferPredLabel, inferPredLogits, inferPredProb, \
                                                mean, std
    
#===============================================================================

if __name__ == '__main__':
    
    classifier = CDclassifier()
    
    trainDir = './train'
    validDir = './valid'
    testDir = './test'
    trialDir = './trial'
    
    #classifier.train( trainDir=trainDir, validDir=validDir )
    classifier.test( testDir=testDir )

    pass

