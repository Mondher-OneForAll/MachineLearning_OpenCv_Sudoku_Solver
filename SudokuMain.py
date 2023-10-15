#### REMOVE TENSORFLOW ERRORS ###########
print('SETTING UP')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#########################################

import cv2
import numpy as np

from utils import *
import SudokuSolver

######################## SETTING ###########################
pathImage = 'Sudoku.jpg'
heightImage = 450
widthImage = 450
model = initializePredictionModel()  # LOAD  THE CNN MODEL
############################################################

######## 1. PREPARE THE IMAGE ###################
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    # img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImage, heightImage))  # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
    imgBlank = np.zeros((heightImage, widthImage, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING :
    imgThreshold = preProcess(img)

    ######## 2. FIND ALL CONTOURS ###################
    imgContours = img.copy()  # COPY IMAGE FOR DISPALY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPALY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)  # DRAW ALL DETECTED CONTOURS

    ######## 3. FIND THE BIGGEST CONTOUR AND USE IT AS SUDOKU ###################
    biggest, maxArea = biggestContour(contours)  # FIND THE BIGGEST CONTOUNR
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 20)  # DRAW THE BIGGEST CONTOUR
        pts1 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0], [widthImage, 0], [0, heightImage], [heightImage, widthImage]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImage, heightImage))
        imgDetectedDigits = imgBlank.copy()
        imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        ######## 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE ###################
        imgSolvedDigits = imgBlank.copy()
        boxes = splitBoxes(imgWarpColored)
        # print(len(boxes))
        # cv2.imshow('Sample', boxes[65])
        numbers = getPrediction(boxes, model)
        '''
            for i in range(len(numbers)):
                if i % 9 == 0 and i != 0:
                    print('')
                print(numbers[i], end=' ')
            '''
        imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        posArray = np.where(numbers > 0, 0, 1)

        ######## 5. FIND SOLUTION OF THE BOARD ###############
        board = np.array_split(numbers, 9)
        try:
            SudokuSolver.solve(board)
        except:
            pass

        flatList = []
        for subList in board:
            for item in subList:
                flatList.append(item)

        solvedNumbers = flatList * posArray
        imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

        ######## 6. OVERLAY SOLUTION ############
        pts2 = np.float32(biggest)  # PREPARE POINTS FOR WARP
        pts1 = np.float32([[0, 0], [widthImage, 0], [0, heightImage], [heightImage, widthImage]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
        imgInvWarpColored = img.copy()
        imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImage, heightImage))
        inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
        imgDetectedDigits = drawGrid(imgDetectedDigits)
        imgSolvedDigits = drawGrid(imgSolvedDigits)

        imgArray = ([img, imgThreshold, imgContours, imgBigContour],
                    [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
        stackedImage = stackImages(imgArray, .7)
        cv2.imshow('Stacked Image', stackedImage)

    else:
        print('No Sudoku Found')

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
