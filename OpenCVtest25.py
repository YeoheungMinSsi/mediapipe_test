import cv2
import mediapipe as mp
import numpy as np
import math


# 각도를 계산하는 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


# 3D 박스를 그리는 함수
def draw_3d_box(image, landmarks, connections):
    image_height, image_width, _ = image.shape
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        start_landmark = landmarks[start_idx]
        end_landmark = landmarks[end_idx]

        start_point = (int(start_landmark.x * image_width), int(start_landmark.y * image_height))
        end_point = (int(end_landmark.x * image_width), int(end_landmark.y * image_height))

        cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    return image


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 이미지 파일 경로를 지정합니다. 실제 이미지 파일 경로로 변경해주세요.
image_path = "img3.jpg"

# 이미지를 읽어옵니다.
image = cv2.imread(image_path)

# 이미지가 None이 아닌지 확인
if image is None:
    print(f"이미지를 읽을 수 없습니다: {image_path}")
    exit()

# 이미지 크기를 500x500으로 조정합니다.
image = cv2.resize(image, (500, 500))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 3D 박스 그리기
        image = draw_3d_box(image, landmarks, mp_pose.POSE_CONNECTIONS)

        # 허리 각도 계산
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        angle = calculate_angle(shoulder, hip, knee)

        # 각도 표시
        cv2.putText(image, f'Waist Angle: {angle:.2f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 자세 판단 및 표시
        if angle > 160:
            posture = "Straight"
            color = (0, 255, 0)  # 녹색
        elif angle > 140:
            posture = "Slightly bent"
            color = (0, 255, 255)  # 노란색
        else:
            posture = "Bent"
            color = (0, 0, 255)  # 빨간색

        cv2.putText(image, f'Posture: {posture}',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # 포즈 랜드마크 그리기
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 결과 이미지를 500x500 픽셀 창에 표시
    cv2.namedWindow('MediaPipe Pose', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MediaPipe Pose', 500, 500)
    cv2.imshow('MediaPipe Pose', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
