import cv2
import mediapipe as mp
import time

class FaceMesh():
    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = False,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode, self.max_num_faces,
                                                 self.refine_landmarks, self.min_detection_confidence,
                                                 self.min_tracking_confidence)
        self.drawSpeck = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)


    def findMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = []
        h, w = img.shape[:2]
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, face_landmarks, self.mpFaceMesh.FACEMESH_CONTOURS,
                                      self.drawSpeck, self.drawSpeck)

                face = []
                for id, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    face.append([x, y])
                faces.append(face)
        return faces

def main():
    cap = cv2.VideoCapture("videos/talkin.mp4")
    detector = FaceMesh()
    pTime = 0
    while True:
        res, img = cap.read()

        lms = detector.findMesh(img)
        for i in range(len(lms)):
            for lm in lms[i]:
                print(f"{lm[0]}, {lm[1]}")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("img", img)
        cv2.waitKey(5)

if __name__ == "__main__":
    main()





