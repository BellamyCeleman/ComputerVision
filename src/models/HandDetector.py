import cv2
import mediapipe as mp
import time

class HandDetector():

    def __init__(self, mode = False, maxHands = 2,
                 detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True):
        markList = []
        # if self.resul
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, landmark in enumerate(myHand.landmark):
                h, w = img.shape[:2]
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                markList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 3, (100, 50, 0), cv2.FILLED)

        return markList

def main():
    pastTime, currentTime = 0, 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        res, img = cap.read()
        img = detector.findHands(img)
        markList = detector.findPosition(img)

        if len(markList) != 0:
            print(markList[4])

        currentTime = time.time()
        fps = 1 / (currentTime - pastTime)
        pastTime = currentTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 50, 0), 2)
        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()

