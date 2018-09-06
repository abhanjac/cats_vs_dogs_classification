#!/usr/bin/env python

from config import *

#===============================================================================

def datasetMeanStd( dataDir=None ):
    '''
    Takes in the location of the images as input.
    Calculates the mean and std of the images of a dataset that is needed 
    to normalize the images before training.
    Returns the mean and std in the form of float arrays 
    (e.g. mean = [ 0.52, 0.45, 0.583 ], std = [ 0.026, 0.03, 0.0434 ] )
    '''
    
    if dataDir == None:
        print( '\ndataDir not provided. Aborting...' )
        return

    listOfImg = os.listdir( dataDir )
    meanOfImg = np.zeros( ( inputImgH, inputImgW, 3 ), dtype=np.float64 )
    meanOfImgSquare = np.zeros( ( inputImgH, inputImgW, 3 ), dtype=np.float64 )
    nImg = len( listOfImg )
    
    for idx, i in enumerate(listOfImg):
        img = cv2.imread( os.path.join( dataDir, i ) )
        
        h, w = img.shape[0], img.shape[1]
        if h > inputImgH or w > inputImgW:  intpol = cv2.INTER_LINEAR
        else:   intpol = cv2.INTER_AREA        
        img = cv2.resize( img, ( inputImgW, inputImgH ), interpolation=intpol )
        
        print( '\rAdding the images to create mean and std {} of {}'.format( idx+1, \
                                                len(listOfImg) ), end = '' )
        meanOfImg += img / nImg
        meanOfImgSquare += img * ( img / nImg )
    
    # Now taking mean of all pixels in the mean image created in the loop.
    # Now meanOfImg is 224x224x3.
    meanOfImg = np.mean( meanOfImg, axis=0 )
    meanOfImgSquare = np.mean( meanOfImgSquare, axis=0 )
    # Now meanOfImg is 224x3.
    meanOfImg = np.mean( meanOfImg, axis=0 )
    meanOfImgSquare = np.mean( meanOfImgSquare, axis=0 )
    # Now meanOfImg is 3.
    
    variance = meanOfImgSquare - meanOfImg * meanOfImg
    std = np.sqrt( variance )
    
    return meanOfImg, std

#===============================================================================

def createBatch( listOfImg=None, batchSize=None, createLabels=True, \
                 shuffle=False, mean=0.0, std=1.0 ):
    '''
    This function takes in a list of images and a batch size and returns an
    image batch, a label batch and the updated listOfImg.
    '''
    if listOfImg == None or batchSize == None:
        print( '\nlistOfImg or batchSize not provided. Aborting...' )
        return

    # Shuffling in place if shuffle flag is True (training phase). 
    # This will be false for test and validation phases.
    if shuffle:    random.shuffle( listOfImg )
    
    listOfBatchImg = listOfImg[ 0 : batchSize ]
    imgBatch = []
    
    for i in listOfBatchImg:
        img = cv2.imread(i)
        h, w = img.shape[0], img.shape[1]
        if h > inputImgH or w > inputImgW:  intpol = cv2.INTER_LINEAR
        else:   intpol = cv2.INTER_AREA
        
        img = cv2.resize( img, ( inputImgW, inputImgH ), interpolation=intpol )
        img = (img - mean) / std   # Converting image to range 0 to 1.
        #img = dataAugment( img )    # Data augmentation. Makes code slow.
        #imgBatch.append( img / 255.0 )
        imgBatch.append( img )
        
    
    #cv2.imshow( 'image', imgBatch[0] )
    #cv2.waitKey(0)
    
    # Removing these from the original listOfImg by first converting them to 
    # set and then removing the set of element in the imgBatch and converting
    # back the resulting set to list.
    listOfImg = list( set( listOfImg ) - set( listOfBatchImg ) )
    
    if createLabels:
        # Takes filepath of an image and extracts label from that using 
        # className2labelIdx dictionary.
        extractLabel = lambda x: className2labelIdx[ re.split( '\.|/| ', x )[-3] ]
        
        labelBatch = [ extractLabel(i) for i in listOfBatchImg ]
    else:
        labelBatch = None
    
    return imgBatch, labelBatch, listOfImg

#===============================================================================

def createTfRecord( fileNameOfTfRec=None, dataLocation=None, createLabels=True, \
                    shuffle=False, mean=0.0, std=1.0 ):
    '''
    This function takes in the location of the folder containing the images and 
    creates a tfrecord file for the images present in that folder.
    '''
    if fileNameOfTfRec == None or dataLocation == None:
        print( '\nfileNameOfTfRec, dataLocation not provided. Aborting...' )
        return

    # This list contains the filepaths for all the images in dataLocation.
    listOfImg = [ os.path.join( dataLocation, i ) for i in os.listdir( dataLocation ) ]
    
    # Shuffling in place if shuffle flag is True (training phase). 
    # This will be false for test and validation phases.
    if shuffle:    random.shuffle( listOfImg )
    
    # Takes filepath of an image and extracts label from that using 
    # className2labelIdx dictionary.
    extractLabel = lambda x: className2labelIdx[ re.split( '\.|/| ', x )[-3] ]
    
    # Creating the list of labels.
    listOfLabel = [ extractLabel(i) for i in listOfImg ]
    
    nSamples = len( listOfImg )
    
    # For running the TFRecordWriter there is no need for a session.
    
    # Defining the tfrecordWriter.
    with tf.python_io.TFRecordWriter( fileNameOfTfRec ) as writer:
    
        # Scanning through all images and labels to create ombined feature object.
        for idx, ( i, l ) in enumerate( zip( listOfImg, listOfLabel ) ):
            img = cv2.imread(i)
            h, w = img.shape[0], img.shape[1]
            if h > inputImgH or w > inputImgW:  intpol = cv2.INTER_LINEAR
            else:   intpol = cv2.INTER_AREA
            
            img = cv2.resize( img, ( inputImgW, inputImgH ), interpolation=intpol )
            img = (img - mean) / std   # Normalizing the image with mean and sd.
            #img = dataAugment( img )    # Data augmentation. Makes code slow.
            #img = img / 255.0         # Converting image to range 0 to 1.
            
            # Creating a bytes feature object for the image.
            img = img.tostring()     # Serializing the image.
            img = tf.compat.as_bytes( img )   # String image to bytes image.
            bytesListImg = tf.train.BytesList( value=[ img ] )
            bytesFeatureImg = tf.train.Feature( bytes_list=bytesListImg )
            
            if createLabels:
                # Creating an int64 feature object for the label.
                int64ListLabel = tf.train.Int64List( value=[ l ] )
                int64FeatureLabel = tf.train.Feature( int64_list=int64ListLabel )
            
                features = tf.train.Features( feature={ 'image': bytesFeatureImg, \
                                                        'label': int64FeatureLabel } )
            else:
                features = tf.train.Features( feature={ 'image': bytesFeatureImg } )
                
            # Creating an example object.
            example = tf.train.Example( features=features )
            
            # Writing the example to the file.
            writer.write( example.SerializeToString() ) 
            
            print( 'Image and label written {}/{}'.format( idx+1, nSamples ) ) 
    
    print( '\nAll images recorded and {} created.\n'.format( fileNameOfTfRec ) )

#===============================================================================

# Pixels should be in range of 0 to 1 and a random value from the range of 
# [ -delta to +delta ] is added to all the pixels.
randomBrightness = lambda img: tf.Session().run( tf.image.random_brightness( \
                                                    img, brightnessDelta ) )

#-------------------------------------------------------------------------------

# Pixels should be in range of 0 to 1 and a random value from the range of 
# [ lower to upper ] (lower and upper should be both +ve values) is added to all
# the pixels.
# For each channel, this operation computes mean of image pixels in the channel 
# and then adjusts each component x of each pixel to 
# (x - mean) * contrast_factor + mean.
randomContrast = lambda img: tf.Session().run( tf.image.random_contrast( \
                                        img, contrastLower, contrastUpper ) )

#-------------------------------------------------------------------------------

# With a 1 in 2 chance, outputs the flipped image (left to right flip).
randomFlipHori = lambda img: tf.Session().run( tf.image.random_flip_left_right( img ) )

#-------------------------------------------------------------------------------

# With a 1 in 2 chance, outputs the flipped image (top to bottom flip).
randomFlipVert = lambda img: tf.Session().run( tf.image.random_flip_up_down( img ) )

#===============================================================================

def dataAugment( img ):
    '''
    Data augmentation function. Randomly selects what kind of data augmentation 
    operation is to be performed on the input image.
    Input image must have all pixels between 0 and 1.
    '''
    
    # Generating the choice of which augmentations to use (randomly).
    nOptions = 4    # Number or augmentation options (we have 4 options).
    choice = np.random.rand(4) - 0.5 > 0     # A boolean array.
    outImg = img

    if choice[0]:   outImg = randomBrightness( outImg )
    if choice[1]:   outImg = randomContrast( outImg )
    if choice[2]:   outImg = randomFlipHori( outImg )
    if choice[3]:   outImg = randomFlipVert( outImg )
    
    return outImg

#===============================================================================

def rename( location=None, categoryName=None, replace=False, startIdx=0 ):
    '''
    If there is a folder of images which needs to be renamed with the name of a
    category and an index, then this function can be used. It takes in 
    the file location and the name of the category and also a flag called replace 
    that indicates whether to replace the original file or make a separate copy.
    '''
    if location == None or categoryName == None:
        print( '\nlocation or categoryName not provided. Aborting...' )
        return

    if not replace:     # If replace is False, then create a copy of the folder.
        newLocation = location + '_renamed'     # Name of new folder.
        
        # If there is already a folder with the same newLocation, then delete it.
        if os.path.exists( newLocation ):       shutil.rmtree( newLocation )
        
        # Now copy the original folder.
        shutil.copytree( location, newLocation )
        
    else:
        newLocation = location
        
    # Now all the files in the newLocation will be renamed.
    listOfFiles = os.listdir( newLocation )    # List of files in the location.
    listOfFiles.sort()       # Sorts in place.
    nFiles = len( listOfFiles )
    
    # Renaming the files one by one.
    for idx, oldFileName in enumerate( listOfFiles ):
        fileExt = oldFileName.split('.')[-1]    # Keep file extension same.
        oldFilePath = os.path.join( newLocation, oldFileName )
        # In some cases we may want the idx of the images to start from 1000 or 50,
        # etc. In that case we use the startIdx.
        newFileName = categoryName + '.' + str( idx + 1 + startIdx ) + '.' + fileExt
        newFilePath = os.path.join( newLocation, newFileName )
        shutil.move( oldFilePath, newFilePath )
        print( 'Renamed file {} to {}, [{}/{}]'.format( oldFileName, \
                                newFileName, idx+1, nFiles ) )
        
#===============================================================================

def timeStamp():
    '''
    Returns the current time stamp including the date and time with as a string 
    of the following format as shown.
    '''
    return datetime.datetime.now().strftime( '_%m_%d_%Y_%H_%M_%S' )

#===============================================================================

# Functions to calculate sigmoid and softmax on numpy arrays.

# Softmax should take in an array of numbers and evaluate the softmax of all 
# the elements of the array and return an array.
softmax = lambda x: np.exp(x) / np.sum( np.exp(x) )

#-------------------------------------------------------------------------------

# Sigmoid should take a single element or an array and evaluate the formula on 
# all of the elements.
sigmoid = lambda x: 1.0 / ( 1.0 + np.exp(-1.0 * x) )

#-------------------------------------------------------------------------------

# Function to normalize input image to the range of 0 to 1
# (provided all the elements dont have the same values, in which case it returns
# the original array).
normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) \
                                if np.max(x) > np.min(x) else x

#-------------------------------------------------------------------------------

# Function to invert a normalized image (all 0s and 1s will be interchanged).
invert = lambda x: 1-x

#===============================================================================

def paramsFromStat( stat ):
    '''
    Function to extract the parameters from the statistics list of a certain epoch.
    The statistics list will have strings for each epoch. Each of these strings 
    will be like the following:
    "epoch, learningRate, trainLoss, trainAcc, validAcc"
    This string will be parsed for getting the parameters.
    To get the parameters for the epoch 4, statistics[4] have to be passes into 
    this function.
    '''
    params = stat.split(', ')
    epoch = int( params[0] )
    learningRate = float( params[1] )
    trainLoss = float( params[2] )
    trainAcc = float( params[3] )
    validAcc = float( params[4] )
    
    return [ epoch, learningRate, trainLoss, trainAcc, validAcc ]

#===============================================================================

def findLatestCkpt( checkpointDirPath=None, training=True ):
    '''
    Finds out the latest checkpoint file in the checkpoint directory and
    deletes the incompletely created checkpoint.
    It returns the metaFilePath and ckptPath if found, else returns None.
    It also returns the epoch number of the latest completed epoch.
    The usual tensorflow funtion used to find the latest checkpoint does not
    take into account the fact that the learning rate or the batch size for the 
    training may have changed since the last checkpoint. So for that this 
    function and the json file created along with the checkpoint are used to 
    find the true latest checkpoint.
    '''
    
    if checkpointDirPath == None:
        print( 'checkpointDirPath not provided... Aborting.' )
        return
    
    # If this is a testing mode, and no checkpoint directory is there, then abort.
    if not os.path.exists( checkpointDirPath ) and not training:
        print( 'checkpoint directory \'{}\' not found... Aborting.'.format( \
                                                        checkpointDirPath ) )
        return

    # Create a folder to store the model checkpoints.
    if not os.path.exists( checkpointDirPath ):  # If no previous model is saved.
        os.makedirs( checkpointDirPath )
        return None, None, 0
    else:
        # If there is previously saved model, then import the graph 
        # along with all the variables, operations etc. (.meta file).
        # Import the variable values (.data binary file with the use of
        # the .index file as well).
        # There is also a file called 'checkpoint' which keeps a record
        # of the latest checkpoint files saved.
        
        # Name of checkpoint to be loaded (in general the latest one).
        # Sometimes due to keyboard interrupts or due to error in saving
        # checkpoints, not all the .meta, .index or .data files are saved.
        # So before loading the checkpoint we need to check if all the 
        # required files are there or not, else the latest complete 
        # checkpoint files should be loaded. And all the incomplete latest 
        # but incomplete ones should be deleted.

        listOfFiles = os.listdir( checkpointDirPath )
        # Remove the 'checkpoint' file first (as we are not using it).
        if 'checkpoint' in listOfFiles:
            listOfFiles.remove( 'checkpoint' )
                
        # List to hold the names of checkpoints which have all files.
        listOfValidCkptPaths = []
        
        # If there are files inside the checkpoint directory.
        while len( listOfFiles ) > 1:
            # Continue till all the files are scanned.
            
            fileName = listOfFiles[-1]
            
            ckptName = '.'.join( fileName.split('.')[:-1] )
            ckptPath = os.path.join( checkpointDirPath, ckptName )
            metaFileName = ckptName + '.meta'
            metaFilePath = ckptPath + '.meta'
            indexFileName = ckptName + '.index'
            indexFilePath = ckptPath + '.index'
            dataFileName = ckptName + '.data-00000-of-00001'
            dataFilePath = ckptPath + '.data-00000-of-00001'
            jsonFileName = ckptName + '.json'
            jsonFilePath = ckptPath + '.json'
            
            if metaFileName in listOfFiles and dataFileName in listOfFiles and \
               indexFileName in listOfFiles and jsonFileName in listOfFiles:
               # All the files exists, so removing them from the listOfFiles 
               # (not deleting them). (Removing all three files together).
                listOfFiles = list( set( listOfFiles ) - set( [ metaFileName, \
                                indexFileName, dataFileName, jsonFileName ] ) )
                listOfValidCkptPaths.append( ckptPath )

            else:
                # If one or more of the .meta, .index or .data files are 
                # missing, then the remaining are deleted and also removed 
                # from the listOfFiles and then we loop back again.
                if os.path.exists( metaFilePath ):
                    os.remove( metaFilePath )
                    listOfFiles.remove( metaFileName )
                if os.path.exists( indexFilePath ):
                    os.remove( indexFilePath )
                    listOfFiles.remove( indexFileName )
                if os.path.exists( dataFilePath ):
                    os.remove( dataFilePath )
                    listOfFiles.remove( dataFileName )
                if os.path.exists( jsonFilePath ):
                    os.remove( jsonFilePath )
                    listOfFiles.remove( jsonFileName )

            #print( len(listOfFiles) )

#-------------------------------------------------------------------------------

        # At this stage we do not have any incomplete checkpoints in the
        # checkpointDirPath. So now we find the latest checkpoint.
        latestCkptIdx, latestCkptPath = 0, None
        for ckptPath in listOfValidCkptPaths:
            currentCkptIdx = ckptPath.split('-')[-1]   # Extract checkpoint index.
            
            # If the current checkpoint index is '', (which can happen if the
            # checkpoints are simple names like yolo_model and do not have 
            # index like yolo_model.ckpt-2 etc.) then break.
            if currentCkptIdx == '':    break
            
            currentCkptIdx = int( currentCkptIdx )
            
            if currentCkptIdx > latestCkptIdx:     # Compare.
                latestCkptIdx = currentCkptIdx
                latestCkptPath = ckptPath
                
        # This will give the latest epoch that has completed successfully.
        # When the checkpoints are saved the epoch is added with +1 in the 
        # filename. So for extracting the epoch the -1 is done.
        latestEpoch = latestCkptIdx if latestCkptIdx > 0 else 0
        
        ##latestCkptPath = tf.train.latest_checkpoint( checkpointDirPath )
        # We do not use the tf.train.latest_checkpoint( checkpointDirPath ) 
        # function here as it is only dependent on the 'checkpoint' file 
        # inside checkpointDirPath. 
        # So this does not work properly if the latest checkpoint mentioned
        # inside this file is deleted because of incompleteness (missing 
        # some files).

        #ckptPath = os.path.join( checkpointDirPath, 'yolo_model.ckpt-0' )

#-------------------------------------------------------------------------------

        if latestCkptPath != None:
            # This will happen when only the 'checkpoint' file remains.
            #print( latestCkptPath )
            latestMetaFilePath = latestCkptPath + '.meta'
            return latestMetaFilePath, latestCkptPath, latestEpoch
        
        else:   
            # If no latest checkpoint is found or all are deleted 
            # because of incompleteness and only the 'checkpoint' file 
            # remains, then None is returned.
            return None, None, 0

#===============================================================================

def createMaskFromContour( imgWidth, imgHeight, contour ):
    '''
    This function takes in the image height and width and the contour and then 
    creates a mask of the same image. This will be useful for evaluating the 
    error as we will try to create the output of the network to be an image that
    is all black, except for the region of the object which will be white.
    It is to see how close the output of the network is to this mask.
    '''
    mask = np.zeros( (imgHeight, imgWidth), dtype=np.uint8 )
    
    # If the contour is a normal list of points where the points are represented 
    # as numpy arrays, then just draw them as they are.
    # Else the points in the list will be represented as lists themselves.
    # In that case convert them to numpy arrays before drawing.
    if type( contour[0] ) == np.ndarray:
        cv2.drawContours( mask, contour, -1, (255), -1 )
    else:
        contourPtsArr = np.array( contour )
        contourPtsArr = contourPtsArr.reshape( -1, 1, 2 )
        cv2.drawContours( mask, [ contourPtsArr ], -1, (255), -1 )
        
    return mask
    
#===============================================================================

def spaceToDepth( arr, blockSize ):
    '''
    Rearranges blocks of spatial data, into depth. More specifically, this 
    operation outputs a copy of the input tensor where values from the height 
    and width dimensions are moved to the depth dimension. The attribute 
    block_size indicates the input block size.

    For the following input of shape [1, 4, 4, 1], and a block size of 2:
    x = [[ [[1],   [2],  [5],  [6]],
           [[3],   [4],  [7],  [8]],
           [[9],  [10], [13],  [14]],
           [[11], [12], [15],  [16]] ]]
    the operator will return the following tensor of shape [1, 2, 2, 4]:
    x = [[ [[1, 2, 3, 4],        [5, 6, 7, 8]],
           [[9, 10, 11, 12], [13, 14, 15, 16]] ]]
    '''
    arr = np.array( arr )
    b, h, w, d = arr.shape
    
    # Height and width should be multiple of blockSize.
    if h % blockSize != 0 or w % blockSize != 0:
        print( 'height or width not multiple of blockSize. Aborting.' )
        return

    newH, newW = int( h / blockSize ), int( w / blockSize )
    arrNew = arr.reshape( b, newH, blockSize, newW, blockSize, -1)
    arrNew = np.swapaxes( arrNew, 2, 3 )
    arrNew = arrNew.reshape( b, newH, newW, -1)
    
    return arrNew
    
#===============================================================================

def depthToSpace( arr, blockSize ):
    '''
    Rearranges data from depth into blocks of spatial data. This is the reverse 
    transformation of SpaceToDepth. More specifically, this op outputs a copy of
    the input tensor where values from the depth dimension are moved in spatial 
    blocks to the height and width dimensions. The attr block_size indicates the
    input block size and how the data is moved.
    
    For the following input of shape [1, 2, 2, 4], and a block size of 2:
    x =  [[ [[1, 2, 3, 4],        [5, 6, 7, 8]],
            [[9, 10, 11, 12], [13, 14, 15, 16]] ]]
    the operator will return the following tensor of shape [1 4 4 1]:
    x = [[ [[1],   [2],  [5],  [6]],
           [[3],   [4],  [7],  [8]],
           [[9],  [10], [13],  [14]],
           [[11], [12], [15],  [16]] ]]
    '''
    arr = np.array( arr )
    b, h, w, d = arr.shape
    
    # Height and width should be multiple of blockSize.
    if d % (blockSize ** 2) != 0:
        print( 'depth not multiple of blockSize^2. Aborting.' )
        return

    newH, newW = h * blockSize, w * blockSize
    arrNew = arr.reshape( b, h, w, blockSize, blockSize, -1 )
    arrNew = np.swapaxes( arrNew, 2, 3 )
    arrNew = arrNew.reshape( b, newH, newW, -1)

    return arrNew
    
#===============================================================================

if __name__ == '__main__':

    trainDir = './train'
    testDir = './test'
    validDir = './valid'
    trialDir = './trial'

    trainImgList = os.listdir( trainDir )
    testImgList = os.listdir( testDir )

##-------------------------------------------------------------------------------

    #trainDogImgList = [ i for i in trainImgList if i.split('.')[0] == 'dog' ]
    #trainCatImgList = [ i for i in trainImgList if i.split('.')[0] == 'cat' ]

    ##os.makedirs( validDir )
    ##for i in range( 10000, len( trainDogImgList ) ):
        ##filename = 'dog.' + str(i) + '.jpg'
        ##shutil.move( os.path.join( trainDir, filename ), os.path.join( validDir, filename ) )
        ##filename = 'cat.' + str(i) + '.jpg'
        ##shutil.move( os.path.join( trainDir, filename ), os.path.join( validDir, filename ) )
        
    #validImgList = os.listdir( validDir )
    
    #validDogImgList = [ i for i in validImgList if i.split('.')[0] == 'dog' ]
    #validCatImgList = [ i for i in validImgList if i.split('.')[0] == 'cat' ]

    #maxHeight, maxWidth, minHeight, minWidth = 0, 0, 10000, 10000
    #avgHeight, avgWidth = 0, 0
    #totalImgs = len( trainImgList ) + len( testImgList ) + len( validImgList )
    
    #for i in trainImgList:
        #filepath = os.path.join( trainDir, i )
        #img = cv2.imread( filepath )
        #h, w = img.shape[0], img.shape[1]
        #maxHeight = h if h > maxHeight else maxHeight
        #maxWidth = w if w > maxWidth else maxWidth
        #minHeight = h if h < minHeight else minHeight
        #minWidth = w if w < minWidth else minWidth
        #avgHeight += (h / totalImgs)
        #avgWidth += (w / totalImgs)
        #print(i)

    #for i in testImgList:
        #filepath = os.path.join( testDir, i )
        #img = cv2.imread( filepath )
        #h, w = img.shape[0], img.shape[1]
        #maxHeight = h if h > maxHeight else maxHeight
        #maxWidth = w if w > maxWidth else maxWidth
        #minHeight = h if h < minHeight else minHeight
        #minWidth = w if w < minWidth else minWidth
        #avgHeight += (h / totalImgs)
        #avgWidth += (w / totalImgs)
        #print(i)
        
    #for i in validImgList:
        #filepath = os.path.join( validDir, i )
        #img = cv2.imread( filepath )
        #h, w = img.shape[0], img.shape[1]
        #maxHeight = h if h > maxHeight else maxHeight
        #maxWidth = w if w > maxWidth else maxWidth
        #minHeight = h if h < minHeight else minHeight
        #minWidth = w if w < minWidth else minWidth
        #avgHeight += (h / totalImgs)
        #avgWidth += (w / totalImgs)
        #print(i)

    #print( 'Total train images: ', len(trainImgList) )
    #print( 'Total \'dog\' images in training set: ', len(trainDogImgList) )
    #print( 'Total \'cat\' images in training set: ', len(trainCatImgList) )
    #print('')
    #print( 'Total validation images: ', len(validImgList) )
    #print( 'Total \'dog\' images in validation set: ', len(validDogImgList) )
    #print( 'Total \'cat\' images in validation set: ', len(validCatImgList) )
    #print( 'Filename format for training and validation images: {}, {}\n[ Printing the 0th image '\
           #'filename of dog and cat ]'.format( trainDogImgList[0], trainCatImgList[0] ) )
    #print('')
    #print( 'Total test images:', len(testImgList) )
    #print( 'Filename format for test images: {}\n[ Printing the 0th image '\
           #'filename ]'.format( testImgList[0] ) )
    #print( 'maxHeight: {}, minHeight: {}, maxWidth: {}, minWidth: {}'.format( \
                                #maxHeight, minHeight, maxWidth, minWidth ) )
    #print( 'avgHeight: {}, avgWidth: {}'.format( avgHeight, avgWidth ) )

    pass
