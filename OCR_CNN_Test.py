import cv2
import numpy as np
import pickle

##################Settings####################
width = 640
height = 480
threshold = 0.65
###############################################

######################### WEBCAM #########################
'''
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
'''
##########################################################

imgOriginal = cv2.imread('resources/6.jpeg')
pickle_in = open('model_trained.p', 'rb')
model = pickle.load(pickle_in)


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


while True:
    # _, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow('Processed Image', img)
    img = img.reshape(1, 32, 32, 1)
    ###Predict###
    classIndex = int(np.argmax(model.predict(img), axis=1))
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    # print(classIndex, probVal)

    if probVal > threshold:
        cv2.putText(imgOriginal, str(classIndex) + '  ' + str(round(probVal * 100)) + '%',
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    imgOriginal = cv2.resize(imgOriginal, (width, height))
    cv2.imshow('Original Image', imgOriginal)

    if cv2.waitKey(1) == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()
