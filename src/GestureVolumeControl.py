import math
from contextlib import nullcontext

import cv2
import mediapipe as mp
import time
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.api.endpointvolume import IAudioEndpointVolume

from src.models.HandDetector import HandDetector
from pycaw.pycaw import AudioUtilities

CAM_WIDTH, CAM_HEIGHT = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(3, CAM_HEIGHT)

pTime = 0
detector = HandDetector(maxHands=1, detectionCon=0.95, trackCon=0.95)



device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

start_time, max_length, observe = None, None, True


while True:
    res, img = cap.read()
    img = detector.findHands(img)

    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        if not start_time:
            start_time = time.time()

        diff = time.time() - start_time

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 5, (255, 0, 0), 2)
        cv2.circle(img, (x2, y2), 5, (255, 0, 0), 2)
        cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0),  thickness=2)

        length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        if diff >= 3 and observe:
            observe = False
            max_length = length
        elif not observe:
            if length > max_length:
                length = max_length
            percentage = length / max_length
            volume.SetMasterVolumeLevelScalar(percentage, None)

        if length < 30:
            cv2.circle(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                       10, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 0), 2)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break