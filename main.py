# Ta koda izračuna število push-up vaj, ki jih oseba izvaja na video posnetku. Za to uporablja biblioteko md_pose,
# ki zaznava pozicijo telesa na posnetku, in cv2, ki omogoča pregledovanje video posnetka.

import cv2
from cvzone.PoseModule import PoseDetector
import mediapipe as md

md_drawing = md.solutions.drawing_utils
md.drawing_styles = md.solutions.drawing_styles

md_pose = md.solutions.pose

count = 0

position = None

cap = cv2.VideoCapture("C:/Users/Vuk/PycharmProjects/pythonPushUpDetetcor/production ID_5195148.mp4")

with md_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("empty camera")
            break
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        result = pose.process(image)

        imList = []
        if result.pose_landmarks:
            md_drawing.draw_landmarks(
                image, result.pose_landmarks, md_pose.POSE_CONNECTIONS)
            for id, im in enumerate(result.pose_landmarks.landmark):
                h, w, _ = image.shape
                X, Y = int(im.x * w), int(im.y * h)
                imList.append([id, X, Y])
        if len(imList) != 0:
            if imList[12][2] and imList[11][2] >= imList[14][2] and imList[13][2]:
                position = "down"
            if (imList[12][2] and imList[11][2] <= imList[14][2] and imList[13][2]) and position == "down":
                position = "up"
                count += 1
                print(count)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(count)
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = int((image.shape[1] - text_size[0]) / 2)
        text_y = int((image.shape[0] + text_size[1]) / 2)
        cv2.putText(image, text, (text_x, text_y), font, 1, (255, 255, 255), 2)

        cv2.imshow("Push-up counter", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
cap.release()
