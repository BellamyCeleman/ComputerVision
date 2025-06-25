import cv2
import mediapipe as mp
import time

class PoseDetector():
    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity,
                            self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation,
                            self.min_detection_confidence, self.min_tracking_confidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if (self.results.pose_landmarks and draw):
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        lmList = []
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 1, (100, 200, 200))
        return lmList

def main():
    cap = cv2.VideoCapture("../../videos/running.mp4")
    pTime = 0
    detector = PoseDetector()
    while True:
        res, img = cap.read()
        img = detector.findPose(img, True)
        lmList = detector.getPosition(img)
        print(lmList[14])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(fps), (50, 50), cv2.FONT_ITALIC, 0.5, (100, 200, 50), 2)

        cv2.imshow("Pic", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

# while True:
#     res, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = pose.process(imgRGB)
#     print(results.pose_landmarks)
#
#     if (results.pose_landmarks):
#         mpDraw.draw_landmarks(img, results.pose_landmarks)
#         for id, lm in enumerate(results.pose_landmarks.landmark):
#             h, w, c = img.shape
#             print(id, lm)
#             cx, cy = int(lm.x * w), int(lm.y * h)
#             cv2.circle(img, (cx, cy), 1, (100, 200, 200))
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#
#     cv2.putText(img, str(fps), (50, 50), cv2.FONT_ITALIC, 0.5, (100, 200, 50), 2)
#
#     cv2.imshow("Pic", img)
#     cv2.waitKey(1)


