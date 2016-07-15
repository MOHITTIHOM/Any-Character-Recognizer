# this program trains our classifier using handwritten DataSet.
# output classifier will be trainedValue.pkl (can be renamed)

import sys
import os
import cv2
import numpy as np
import operator
from sklearn.externals import joblib
from sklearn import datasets

from sklearn.svm import LinearSVC
from sklearn import preprocessing


contourArea = 100
imageWidth = 20
imageHeight = 30

#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

def main():
    path="dataset"
#####    The ML approach is to use a linear SVM    #########################################################
    clf = LinearSVC()
	
	
#####    xValues are labels   ###############################################################################
    xValue=[]

#####    yValues are features   ###############################################################################
    yValue_Images =  np.empty((0, imageWidth * imageHeight))
	
	
	
    for x in xrange(65,91):
        trainingImages = cv2.imread(os.path.join(path,chr(x)+'.png'))

########   image doesnt exist          #######################################################################
        if trainingImages is None:
            print "\n\n File doesn't exist \n\n"
            os.system("pause")
            return
       
########   image processing for better evaluation #############################################################
        imgGray = cv2.cvtColor(trainingImages, cv2.COLOR_BGR2GRAY)
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)


        imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)


########    make a copy to obtain contours ####################################################################
        imgThreshCopy = imgThresh.copy()

########    cv2 functions to obtain various contours ##########################################################
        imgCount, imageContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


        for imageContour in imageContours:
            if cv2.contourArea(imageContour) > contourArea:
                [intX, intY, intW, intH] = cv2.boundingRect(imageContour)


                cv2.rectangle(trainingImages,(intX, intY),(intX+intW,intY+intH),(0, 0, 255),2)

                tempImg = imgThresh[intY:intY+intH, intX:intX+intW]
                resizedTemp = cv2.resize(tempImg, (imageWidth, imageHeight))
                

                xValue.append(x)

                resizedImage = resizedTemp.reshape((1, imageWidth * imageHeight))
                yValue_Images = np.append(yValue_Images, resizedImage, 0)




    flattenedXValues = np.array(xValue, np.float32)

    

    print "\n\n Done \n\n"
    labels = flattenedXValues.reshape((flattenedXValues.size,1))
    features = yValue_Images
	
#############    train SVM classifier   #############################################################################
    clf.fit(features, labels.ravel())

######   save the trained value  ####################################################################################
    joblib.dump((clf), "trainedValue.pkl", compress=3)
    

    return

#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
if __name__ == "__main__":
    main()
