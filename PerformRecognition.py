# this program performs recognition using the trained class.
# note the output is horizontally sorted


import cv2
import numpy as np
import operator
import os
from sklearn.externals import joblib

contourArea = 100
imageWidth = 20
imageHeight = 30

# Object specifications : the following is used to describe the contour and get valid contours afterwards
class contourData():
	imageContour = None
	boundingRect = None
	rectX = 0
	rectY = 0
	rectWidth = 0
	rectHeight = 0
	contourAREA = 0.0

	def calculateData(self):
		[X,Y,width,height] = self.boundingRect
		self.rectX = X
		self.rectY = Y
		self.rectWidth = width
		self.rectHeight = height

	def contourIsValid(self):
		if self.contourAREA < contourArea: return False
		return True
########   class definition ends here   #####################################################################



def main():
########    next we load the trained class here  #############################################################
    clf = joblib.load("trainedValue.pkl")

########    load the image to test     #######################################################################
    trainingImages = cv2.imread("test.png")

########   image doesnt exist          #######################################################################
    if trainingImages is None:
            print "\n\n File doesn't exist \n\n"
            os.system("pause")
            return

########   image processing for better evaluation #############################################################
    imgGray = cv2.cvtColor(trainingImages, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)


    imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

########    make a copy to obatin contours ####################################################################
    imgThreshCopy = imgThresh.copy()

########    cv2 functions to obtain various contours ##########################################################
    imgCount, imageContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

######## till now we have obtained various contours in random order and shape
######## following steps are
######## 1) we will assign them to objects (which is very important as we want to have them sorted horizontally)
######## 2) find valid contours in them
######## 3) sort all the valid contours horizontally and
######## 4) append the predicted value for contour to the predictString

#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#



########              array of all contours          ##########################################################
    contourArray = []
########              array of all validContour      ##########################################################
    validContour = []

#~#~#~#~#~#~##~#~#~#~#           STEP-1:              #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

########   make object for all the contours and add them to contourArray   ####################################
    for imageContour in imageContours:
        contourDATA = contourData()
        contourDATA.imageContour = imageContour
        contourDATA.boundingRect = cv2.boundingRect(contourDATA.imageContour)
        contourDATA.calculateData()
        contourDATA.contourAREA = cv2.contourArea(contourDATA.imageContour)
        contourArray.append(contourDATA)

#~#~#~#~#~#~##~#~#~#~#           STEP-2:              #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

########   find all the valid contours and add them to valid contours       ####################################
    for contourDATA in contourArray:
        if contourDATA.contourIsValid():
            validContour.append(contourDATA)

#~#~#~#~#~#~##~#~#~#~#           STEP-3:              #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#

########   sort the valid contours with respect to X or horizontally       ####################################
    validContour.sort(key =  operator.attrgetter("rectX"))
########   you may sort it vertically as well    ##############################################################
	
	
    predictedString = ""




#~#~#~#~#~#~##~#~#~#~#           STEP-4:              #~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
    for contourDATA in validContour:
        cv2.rectangle(trainingImages,(contourDATA.rectX,contourDATA.rectY),(contourDATA.rectX + contourDATA.rectWidth, contourDATA.rectY + contourDATA.rectHeight),(0,255,0),2)

        tempImg = imgThresh[contourDATA.rectY: contourDATA.rectY + contourDATA.rectHeight,contourDATA.rectX: contourDATA.rectX + contourDATA.rectWidth]

        resizedTemp = cv2.resize(tempImg, (imageWidth, imageHeight))

        toPredict = resizedTemp.reshape((1, imageHeight * imageWidth))

        toPredict = np.float32(toPredict)

########  predict the value of the contour   ###################################################################
        nbr = clf.predict(toPredict)

        tempOutput = str(unichr(int(nbr)))
########  append the output to the answer   ###################################################################
        predictedString = predictedString + tempOutput



    print "\n" + predictedString + "\n"

    cv2.imshow("Testing Numbers", trainingImages)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return

#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
if __name__ == "__main__":
    main()










