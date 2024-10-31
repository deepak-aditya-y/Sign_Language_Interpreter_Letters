import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
OS = 20
IS = 300
folder = "ALPHA/D"
count = 0

while True:
    success, img = cap.read()
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

        else:
            k = IS / w
            hCal = math.ceil(h * k)
            imgResize = cv2.resize(imgCrop, (IS, hCal))
            imgShape = imgResize.shape
            hGap = math.ceil((IS - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(count)
