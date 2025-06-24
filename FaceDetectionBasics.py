import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, min_detection_confidence: float = 0.5,
                 model_selection: int = 0 ):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            self.min_detection_confidence,
            self.model_selection
        )

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process((imgRGB))
        bboxs = []

        if  self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # print(id, detection)
                # print(f"Score: {detection.score}")
                print(f"location: {detection.location_data.relative_bounding_box}")
                # print(f"label_id: {detection.label_id}")



                # mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
                bboxs.append(bbox)
                x, y, w_box, h_box = bbox
                cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (200, 200, 200), 2)
                cv2.putText(img, f"{round(detection.score[0] * 100, 2)}%", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return bboxs

    def draw(self, img, bbox, length: int):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.line(img, (x, y), (x + length, y), (200, 200, 200), 7)
        cv2.line(img, (x, y), (x, y + length), (200, 200, 200), 7)

        cv2.line(img, (x1, y), (x1 - length, y), (200, 200, 200), 7)
        cv2.line(img, (x1, y), (x1, y + length), (200, 200, 200), 7)

        cv2.line(img, (x, y1), (x + length, y1), (200, 200, 200), 7)
        cv2.line(img, (x, y1), (x, y1 - length), (200, 200, 200), 7)

        cv2.line(img, (x1, y1), (x1 - length, y1), (200, 200, 200), 7)
        cv2.line(img, (x1, y1), (x1, y1 - length), (200, 200, 200), 7)
        return img

def main():
    cap = cv2.VideoCapture("videos/talkin.mp4")
    pTime = 0
    detector = FaceDetector()

    while True:
        res, img = cap.read()
        coords = detector.findFaces(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        for bbox in coords:
            img = detector.draw(img, bbox, 30)

        cv2.imshow("img", img)
        cv2.waitKey(5)

if __name__ == "__main__":
    main()




