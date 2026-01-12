import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"

# Ensure images load in correct order
myList = sorted(os.listdir(folderPath), key=lambda x: int(x.split('.')[0]))
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

pTime = 0
detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList, handType = detector.findPosition(img, draw=False)

    if len(lmList) != 0 and handType is not None:
        fingers = []

        # 👍 Thumb
        if handType == "Right":
            fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)
        else:
            fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)

        # ☝️ Other fingers
        for id in range(1, 5):
            fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)

        totalFingers = fingers.count(1)

        # 🖼️ IMAGE MAPPING
        # 0 fingers → 6.jpeg (index 5)
        if totalFingers == 0:
            overlayImg = overlayList[5]
        else:
            overlayImg = overlayList[totalFingers - 1]

        h, w, c = overlayImg.shape
        img[0:h, 0:w] = overlayImg

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
