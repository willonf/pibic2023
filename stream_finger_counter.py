import cv2
import mediapipe as mp

camera_capture = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Possui parâmetros de ajustes
mp_draw = mp.solutions.drawing_utils

while True:
    check, img = camera_capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    hands_points = results.multi_hand_landmarks
    height, weight, _ = img.shape
    points = []

    if hands_points:
        for point in hands_points:
            mp_draw.draw_landmarks(img, point, mp_hands.HAND_CONNECTIONS)
            for id, coordinate in enumerate(point.landmark):
                coord_x, coord_y = int(coordinate.x * weight), int(coordinate.y * height)
                points.append((coord_x, coord_y))
                # cv2.putText(img, str(id), (coord_x, coord_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        fingers = [8, 12, 16, 20]
        counter = 0
        if points:
            if points[4][0] < points[2][0]:
                counter += 1
            for x in fingers:
                if points[x][1] < points[x - 2][1]:
                    counter += 1
        cv2.rectangle(img, (80, 10), (200, 100), (255, 255, 0), -1)
        cv2.putText(img, str(counter), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
