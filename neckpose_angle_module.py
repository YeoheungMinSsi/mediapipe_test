import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageFont, ImageDraw
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def create_midpoint(point1, point2):
    return type('Landmark', (), {
        'x': (point1.x + point2.x) / 2,
        'y': (point1.y + point2.y) / 2,
        'z': (point1.z + point2.z) / 2
    })

def calculate_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def draw_text_korean(image, text, position, font_path, font_size, color):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def get_pose_angle():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    angles = {'neck_time': [], 'neck_angle': []}
    elapsed_time = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                continue

            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]

                shoulder_center = create_midpoint(left_shoulder, right_shoulder)
                head_center = create_midpoint(left_ear, right_ear)

                image_height, image_width, _ = image.shape

                vertical_point = type('Landmark', (), {'x': shoulder_center.x, 'y': 0})

                cv2.line(image,
                         (int(shoulder_center.x * image_width), int(shoulder_center.y * image_height)),
                         (int(shoulder_center.x * image_width), 0),
                         (255, 0, 0), 2)

                cv2.line(image,
                         (int(shoulder_center.x * image_width), int(shoulder_center.y * image_height)),
                         (int(head_center.x * image_width), int(head_center.y * image_height)),
                         (0, 255, 0), 2)

                angle = calculate_angle(vertical_point, shoulder_center, head_center)
                lean_direction = "오른쪽" if head_center.x < shoulder_center.x else "왼쪽"

                angle_text = f'기울어진 각도: {angle:.1f}도 ({lean_direction})'
                font_path = "NanumGothic.ttf"
                font_size = 30
                color = (255, 255, 255)
                image = draw_text_korean(image, angle_text, (10, 30), font_path, font_size, color)

                if angle > 10:
                    cv2.line(image,
                             (int(shoulder_center.x * image_width), int(shoulder_center.y * image_height)),
                             (int(head_center.x * image_width), int(head_center.y * image_height)),
                             (0, 0, 255), 3)

                if time.time() - start_time >= 10:  # 10초마다 각도 저장
                    elapsed_time += 10
                    rounded_angle = round(angle, 1)  # 소수점 첫 번째 자리까지 반올림
                    angles['neck_time'].append(elapsed_time)
                    angles['neck_angle'].append(rounded_angle)
                    start_time = time.time()

            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    return angles