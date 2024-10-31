import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classify = Classifier("Model/keras_model.h5", "Model/labels.txt")

OS = 20
IS = 300
folder = "ALPHA/C"
count = 0
Lab = ["A", "B", "C"]
while True:
    success, img = cap.read()
    img2 = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((IS, IS, 3), np.uint8) * 255
        imgCrop = img[y - OS:y + h + OS, x - OS:x + w + OS]

        AS = h / w
        if AS > 1:
            k = IS / h
            wCal = math.ceil(w * k)
            imgResize = cv2.resize(imgCrop, (wCal, IS))
            imgShape = imgResize.shape
            wGap = math.ceil((IS - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classify.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = IS / w
            hCal = math.ceil(h * k)
            imgResize = cv2.resize(imgCrop, (IS, hCal))
            imgShape = imgResize.shape
            hGap = math.ceil((IS - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classify.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        cv2.putText(img2, Lab[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(img2, (x - OS, y - OS), (x + w + OS, y + h + OS), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img2)
    cv2.waitKey(1)
