import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def create_midpoint(point1, point2):
    return type('Landmark', (), {
        'x': (point1.x + point2.x) / 2,
        'y': (point1.y + point2.y) / 2,
        'z': (point1.z + point2.z) / 2
    })


cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 키포인트 생성
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # 0번 키포인트 (머리 중앙)
            head_center = nose

            # 1번 키포인트 (목)
            neck_point = create_midpoint(left_shoulder, right_shoulder)
            neck_point.x = neck_point.x + 0.03

            # 2번 키포인트 (척추 중앙)
            spine_center = create_midpoint(neck_point, create_midpoint(left_hip, right_hip))
            # 척추 쪽으로 조정
            spine_center.y = spine_center.y - 0.05  # 위치를 약간 위로 조정
            spine_center.x = spine_center.x + 0.03  # x위치 앞뒤의 값을 조정 +가 등쪽 -가 배쪽

            # 3번 키포인트 (오른쪽 어깨)
            right_shoulder_point = right_shoulder

            # 4번 키포인트 (왼쪽 어깨)
            left_shoulder_point = left_shoulder

            # 5번 키포인트 (오른쪽 엉덩이)
            right_hip_point = right_hip

            # 6번 키포인트 (왼쪽 엉덩이)
            left_hip_point = left_hip

            # 7번 키포인트 (꼬리뼈)
            tailbone_point = create_midpoint(left_hip, right_hip)
            # 꼬리뼈 쪽으로 조정
            tailbone_point.y = tailbone_point.y + 0.05  # 위치를 약간 아래로 조정

            # 8번 키포인트 (왼쪽 무릎)
            left_knee_point = left_knee

            # 9번 키포인트 (오른쪽 무릎)
            right_knee_point = right_knee

            # 10번 키포인트 (왼쪽 발목)
            left_ankle_point = left_ankle

            # 11번 키포인트 (오른쪽 발목)
            right_ankle_point = right_ankle

            # 12번 키포인트 (왼쪽 팔꿈치)
            left_elbow_point = left_elbow

            # 13번 키포인트 (오른쪽 팔꿈치)
            right_elbow_point = right_elbow

            # 포인트와 연결 그리기
            image_height, image_width, _ = image.shape

            # 연결선 그리기
            connections = [
                (head_center, neck_point), (neck_point, spine_center), (spine_center, tailbone_point),
                (neck_point, right_shoulder_point), (neck_point, left_shoulder_point),
                (right_shoulder_point, right_elbow_point), (left_shoulder_point, left_elbow_point),
                (right_hip_point, tailbone_point), (left_hip_point, tailbone_point),
                (left_hip_point, left_knee_point), (right_hip_point, right_knee_point),
                (left_knee_point, left_ankle_point), (right_knee_point, right_ankle_point)
            ]

            for connection in connections:
                start_point = connection[0]
                end_point = connection[1]
                cv2.line(image,
                         (int(start_point.x * image_width), int(start_point.y * image_height)),
                         (int(end_point.x * image_width), int(end_point.y * image_height)),
                         (0, 255, 0), 2)

            # 키포인트 그리기 및 번호 표시
            points = [
                head_center, neck_point, spine_center, right_shoulder_point, left_shoulder_point,
                right_hip_point, left_hip_point, tailbone_point, left_knee_point, right_knee_point,
                left_ankle_point, right_ankle_point, left_elbow_point, right_elbow_point
            ]
            for i, point in enumerate(points):
                x = int(point.x * image_width)
                y = int(point.y * image_height)
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(image, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()