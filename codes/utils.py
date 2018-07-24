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
        #imgBatch.append( img / 255.0 )
        imgBatch.append( img )
        
    
    #cv2.imshow( 'image', imgBatch[0] )
    #cv2.waitKey(0)
    
    # Removing these from the original listOfImg by first converting them to 
    # set and then removing the set of element in the imgBatch and converting
    # back the resulting set to list.
    listOfImg = list( set( listOfImg ) - set( listOfBatchImg ) )
    
    if createLabels:
        # If this batch is created for training or validation set, only then 
        # labels are needed. Labels may not be needed during testing (and it 
        # may not be possible to extract labels from test filenames either.

        # Takes filepath of an image and extracts label from that using 
        # className2labelIdx dictionary.
        extractLabel = lambda x: className2labelIdx[ re.split( '\.|/| ', x )[-3] ]
        
        labelBatch = [ extractLabel(i) for i in listOfBatchImg ]
    else:
        labelBatch = None
    
    return imgBatch, labelBatch, listOfImg

#===============================================================================

def rename( location=None, categoryName=None, replace=False ):
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
    sorted( listOfFiles )       # Files are sorted in place before renaming.
    nFiles = len( listOfFiles )
    
    # Renaming the files one by one.
    for idx, oldFileName in enumerate( listOfFiles ):
        oldFilePath = os.path.join( newLocation, oldFileName )
        newFileName = categoryName + '.' + str( idx + 1 ) + '.jpg'
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
        return None, None, -1
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
        latestCkptIdx, latestCkptPath = -1, None
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
        latestEpoch = latestCkptIdx - 1 if latestCkptIdx > -1 else -1
        
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
            return None, None, -1

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

if __name__ == '__main__':

    trainDir = './train'
    testDir = './test'
    validDir = './valid'
    trialDir = './trial'

    trainImgList = os.listdir( trainDir )
    testImgList = os.listdir( testDir )

#-------------------------------------------------------------------------------

    trainDogImgList = [ i for i in trainImgList if i.split('.')[0] == 'dog' ]
    trainCatImgList = [ i for i in trainImgList if i.split('.')[0] == 'cat' ]

    if not os.path.exists( validDir ):
        os.makedirs( validDir )
        for i in range( 10000, len( trainDogImgList ) ):
            filename = 'dog.' + str(i) + '.jpg'
            shutil.move( os.path.join( trainDir, filename ), os.path.join( validDir, filename ) )
            filename = 'cat.' + str(i) + '.jpg'
            shutil.move( os.path.join( trainDir, filename ), os.path.join( validDir, filename ) )
        
    validImgList = os.listdir( validDir )
    
    validDogImgList = [ i for i in validImgList if i.split('.')[0] == 'dog' ]
    validCatImgList = [ i for i in validImgList if i.split('.')[0] == 'cat' ]

    maxHeight, maxWidth, minHeight, minWidth = 0, 0, 10000, 10000
    avgHeight, avgWidth = 0, 0
    totalImgs = len( trainImgList ) + len( testImgList ) + len( validImgList )
    
    for i in trainImgList:
        filepath = os.path.join( trainDir, i )
        img = cv2.imread( filepath )
        h, w = img.shape[0], img.shape[1]
        maxHeight = h if h > maxHeight else maxHeight
        maxWidth = w if w > maxWidth else maxWidth
        minHeight = h if h < minHeight else minHeight
        minWidth = w if w < minWidth else minWidth
        avgHeight += (h / totalImgs)
        avgWidth += (w / totalImgs)
        print(i)

    for i in testImgList:
        filepath = os.path.join( testDir, i )
        img = cv2.imread( filepath )
        h, w = img.shape[0], img.shape[1]
        maxHeight = h if h > maxHeight else maxHeight
        maxWidth = w if w > maxWidth else maxWidth
        minHeight = h if h < minHeight else minHeight
        minWidth = w if w < minWidth else minWidth
        avgHeight += (h / totalImgs)
        avgWidth += (w / totalImgs)
        print(i)
        
    for i in validImgList:
        filepath = os.path.join( validDir, i )
        img = cv2.imread( filepath )
        h, w = img.shape[0], img.shape[1]
        maxHeight = h if h > maxHeight else maxHeight
        maxWidth = w if w > maxWidth else maxWidth
        minHeight = h if h < minHeight else minHeight
        minWidth = w if w < minWidth else minWidth
        avgHeight += (h / totalImgs)
        avgWidth += (w / totalImgs)
        print(i)

    print( 'Total train images: ', len(trainImgList) )
    print( 'Total \'dog\' images in training set: ', len(trainDogImgList) )
    print( 'Total \'cat\' images in training set: ', len(trainCatImgList) )
    print('')
    print( 'Total validation images: ', len(validImgList) )
    print( 'Total \'dog\' images in validation set: ', len(validDogImgList) )
    print( 'Total \'cat\' images in validation set: ', len(validCatImgList) )
    print( 'Filename format for training and validation images: {}, {}\n[ Printing the 0th image '\
           'filename of dog and cat ]'.format( trainDogImgList[0], trainCatImgList[0] ) )
    print('')
    print( 'Total test images:', len(testImgList) )
    print( 'Filename format for test images: {}\n[ Printing the 0th image '\
           'filename ]'.format( testImgList[0] ) )
    print( 'maxHeight: {}, minHeight: {}, maxWidth: {}, minWidth: {}'.format( \
                                maxHeight, minHeight, maxWidth, minWidth ) )
    print( 'avgHeight: {}, avgWidth: {}'.format( avgHeight, avgWidth ) )

