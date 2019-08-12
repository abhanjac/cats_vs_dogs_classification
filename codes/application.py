#!/usr/bin/env python

from config import *
from utils import *
from train_classifier import *

#===============================================================================

if __name__ == '__main__':
    
    classifier = CDclassifier()
    
    trainDir = './train'
    validDir = './valid'
    testDir = './test'
    trialDir = './trial'
    
#-------------------------------------------------------------------------------

    # Checking with the images of the trial directory.
    inferDir = trialDir
    
    listOfInferImg = os.listdir( inferDir )
    nImgs = len( listOfInferImg )
    listOfInferImg.sort()
    key = ord('`')
    i = int( nImgs / 2 )

    while key & 0xFF != 27:
        # Read image.
        img = cv2.imread( os.path.join( inferDir, listOfInferImg[i] ) )
        imgH, imgW = img.shape[0], img.shape[1]
        
        # Evaluate inference.
        inferLayerOut, inferPredLabel, inferPredLogits, inferPredProb, \
                                mean, std = classifier.inference( img )

        # inferPredLabel is the predicted label index.
        predictedLabelName = classNames[ inferPredLabel ]
        
#-------------------------------------------------------------------------------

        # Taking note of the kernel and bias weights between the GMP and fully
        # connected layer.
        gmp2FcW = classifier.getWeight( 'model/dense8/kernel' )
        gmp2FcB = classifier.getWeight( 'model/dense8/bias' )

        # Combining the channels of the GMP layer.
        _, gmpH, gmpW, gmpC = inferLayerOut['activation7'].shape

#-------------------------------------------------------------------------------

        # We will be looking into raw conv output and output via activation
        # layer after the activation, before the gmp layer.
        combinedChannelConv = np.zeros( (gmpH, gmpW) )
        
        for j in range( gmpC ):
            convOutput = inferLayerOut['conv7'][ :, :, :, j ]
            
            convOutput = np.reshape( convOutput, (gmpH, gmpW) )

            w = gmp2FcW[ j, inferPredLabel ]
            b = gmp2FcB[ inferPredLabel ]
            
            # Now normalizing the outputs.
            # This makes combinedChannelConv image more clear and smooth.
            combinedChannelConv = normalize( convOutput )
            
            combinedChannelConv += convOutput * w + b
        
#-------------------------------------------------------------------------------

        # This combinedChannel should be normalized now between 0 to 1.
        # This is IMPORTANT otherwise the combinedChannel output will be garbage.
        # After that we have to scale it to 255 for displaying.
        combinedChannelConv = normalize( combinedChannelConv ) * 255
        
        # Converting the combined channels into int else there are errors.
        combinedChannelConv = np.asarray( combinedChannelConv, dtype=np.uint8 )
        
        combinedChannelConv = cv2.resize( combinedChannelConv, \
                                    (imgW, imgH), interpolation=cv2.INTER_LINEAR )
        
#-------------------------------------------------------------------------------
        
        # Colormapping the combinedChannel image.
        combinedChannelConv = cv2.applyColorMap( combinedChannelConv, \
                                                    cv2.COLORMAP_JET )
        
        # Overlapping the combinedChannel over the input img.
        combinedChannelConvImposed = combinedChannelConv * 0.5 + img * 0.5
        combinedChannelConvImposed = np.asarray( combinedChannelConvImposed, \
                                                        dtype=np.uint8 )
        
        cv2.imshow( 'gmp filter (conv)', combinedChannelConv )
        cv2.imshow( 'gmp filter (conv imposed)', combinedChannelConvImposed )
        
#-------------------------------------------------------------------------------

        cv2.putText( img, predictedLabelName, (30, 30), \
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA )
        
        ## Showing a combined image.
        #resized = np.hstack( ( img, combinedChannelConvImposed ) )
        #resized = cv2.resize( resized, (448, 224), interpolation = cv2.INTER_AREA )
        #cv2.imshow( 'Image and gmp filter (conv)', resized )
        ##cv2.imwrite( 'combined_' + listOfInferImg[i], resized )
        
        print( 'Actual Label: {}, Predicted Label: {}'.format( \
                    listOfInferImg[i], predictedLabelName ) )
        
#-------------------------------------------------------------------------------

        # Actual image and prediction.
        cv2.imshow( 'Image', img )
        
        key = cv2.waitKey( 0 )
        if key == 81: i -= 1    # Previous image.
        if key == 83: i += 1    # Next image.
        if i >= nImgs:   i = i % nImgs   # Start over when all images scanned.


    pass





        

