import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageFont, ImageDraw
import time
import csv
import os
from gtts import gTTS
from playsound import playsound

# 기본 디렉토리 및 폰트 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, 'static', 'fonts', 'NanumGothic.ttf')

# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 사진 저장 경로 설정
photo_save_dir1 = 'back_images'

# 디렉토리가 없으면 생성
os.makedirs(photo_save_dir1, exist_ok=True)

# 기존 파일 삭제 (디렉토리가 존재할 경우에만)
if os.path.exists(photo_save_dir1):
    for filename in os.listdir(photo_save_dir1):
        file_path = os.path.join(photo_save_dir1, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted photo: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


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


def draw_text_korean(image, text, position, font_size, color):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def get_pose_angle(save_interval=2, photo_save_interval=2):
    cap = cv2.VideoCapture(0)
    is_recording = False
    is_paused = False
    start_time = 0
    pause_start_time = 0
    total_pause_time = 0
    current_angle = 0
    csv_filename = 'back_angles.csv'
    saved_data = []
    elapsed_time = 0
    last_save_time = 0
    last_photo_save_time = 0

    # CSV 파일 초기화
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['back_time', 'back_angle'])

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("카메라를 찾을 수 없습니다.")
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                shoulder_center = create_midpoint(left_shoulder, right_shoulder)
                hip_center = create_midpoint(left_hip, right_hip)

                image_height, image_width, _ = image.shape

                vertical_point = type('Landmark', (), {'x': hip_center.x, 'y': shoulder_center.y})

                cv2.line(image,
                         (int(hip_center.x * image_width), int(hip_center.y * image_height)),
                         (int(hip_center.x * image_width), 0),
                         (255, 0, 0), 2)

                cv2.line(image,
                         (int(hip_center.x * image_width), int(hip_center.y * image_height)),
                         (int(shoulder_center.x * image_width), int(shoulder_center.y * image_height)),
                         (0, 255, 0), 2)

                angle = calculate_angle(vertical_point, hip_center, shoulder_center)
                current_angle = round(angle, 1)
                lean_direction = "오른쪽" if shoulder_center.x > hip_center.x else "왼쪽"

                angle_text = f'기울기 각도: {current_angle:.1f}도 {lean_direction} 방향'
                font_size = 30
                color = (255, 255, 255)
                image = draw_text_korean(image, angle_text, (10, 30), font_size, color)

                if angle > 5:
                    cv2.line(image,
                             (int(hip_center.x * image_width), int(hip_center.y * image_height)),
                             (int(shoulder_center.x * image_width), int(shoulder_center.y * image_height)),
                             (0, 0, 255), 3)

            # 경과 시간 표시
            if is_recording:
                if not is_paused:
                    elapsed_time = int(time.time() - start_time - total_pause_time)
                time_text = f'경과 시간: {elapsed_time}초'
                if is_paused:
                    time_text += ' (일시정지)'
                image = draw_text_korean(image, time_text, (10, 70), font_size, color)

            # 버튼 추가
            cv2.rectangle(image, (10, image_height - 60), (110, image_height - 10), (0, 255, 0), -1)
            cv2.rectangle(image, (120, image_height - 60), (220, image_height - 10), (0, 0, 255), -1)
            cv2.rectangle(image, (230, image_height - 60), (330, image_height - 10), (255, 255, 0), -1)
            image = draw_text_korean(image, "시작" if not is_recording else "기록 중", (20, image_height - 50), 20,
                                     (0, 0, 0))
            image = draw_text_korean(image, "정지", (130, image_height - 50), 20, (0, 0, 0))
            image = draw_text_korean(image, "일시정지" if not is_paused else "재개", (240, image_height - 50), 20, (0, 0, 0))

            cv2.imshow('MediaPipe Pose', image)

            # 마우스 클릭 이벤트 처리
            def mouse_callback(event, x, y, flags, param):
                nonlocal is_recording, is_paused, start_time, elapsed_time, last_save_time, total_pause_time, pause_start_time
                if event == cv2.EVENT_LBUTTONDOWN:
                    if 10 <= x <= 110 and image_height - 60 <= y <= image_height - 10:
                        if not is_recording:
                            is_recording = True
                            start_time = time.time()
                            last_save_time = 0
                            total_pause_time = 0
                            print("기록 시작")
                        elif is_paused:
                            is_paused = False
                            total_pause_time += time.time() - pause_start_time
                            print("기록 재개")
                    elif 120 <= x <= 220 and image_height - 60 <= y <= image_height - 10:
                        if is_recording:
                            is_recording = False
                            is_paused = False
                            total_pause_time = 0
                            print("기록 정지")
                            print("저장된 값:")
                            for data in saved_data:
                                print(f"시간: {data[0]}초, 각도: {data[1]}도")
                    elif 230 <= x <= 330 and image_height - 60 <= y <= image_height - 10:
                        if is_recording and not is_paused:
                            is_paused = True
                            pause_start_time = time.time()
                            print("일시정지")
                        elif is_recording and is_paused:
                            is_paused = False
                            total_pause_time += time.time() - pause_start_time
                            print("기록 재개")

            cv2.setMouseCallback('MediaPipe Pose', mouse_callback)

            # 각도 저장 및 평균 각도 계산
            if is_recording and not is_paused and elapsed_time >= last_save_time + save_interval:
                saved_data.append((elapsed_time, current_angle))
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([elapsed_time, current_angle])
                print(f"값 저장: 시간 {elapsed_time}초, 각도 {current_angle}도")

                # 10초마다 평균 각도 계산 및 출력
                if elapsed_time % 10 == 0:
                    last_10_angles = [angle for time, angle in saved_data if elapsed_time - 10 < time <= elapsed_time]
                    if last_10_angles:
                        average_angle = np.mean(last_10_angles)
                        print(f"{elapsed_time}초까지의 평균 각도: {average_angle:.2f}도")

                        # 평균 각도가 1도 이상일 때 TTS로 경고 메시지 출력
                        if average_angle >= 1:
                            alert_filename = f"alert_{int(time.time())}.mp3"
                            tts = gTTS("허리를 펴세요", lang='ko')
                            tts.save(alert_filename)
                            playsound(alert_filename)
                            os.remove(alert_filename)

                last_save_time = elapsed_time

            # 사진 저장
            if is_recording and not is_paused and elapsed_time >= last_photo_save_time + photo_save_interval:
                os.makedirs(photo_save_dir1, exist_ok=True)
                photo_filename = os.path.join(photo_save_dir1, f"photo_{int(time.time())}.jpg")
                cv2.imwrite(photo_filename, image)
                print(f"사진 저장: {photo_filename}")
                last_photo_save_time = elapsed_time

            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC 키로 종료
                print("프로그램 종료")
                print("저장된 모든 값:")
                for data in saved_data:
                    print(f"시간: {data[0]}초, 각도 {data[1]}도")

                # 프로그램 종료 시 사진 파일 삭제
                for filename in os.listdir(photo_save_dir1):
                    file_path = os.path.join(photo_save_dir1, filename)
                    os.remove(file_path)
                    print(f"Deleted photo: {file_path}")

                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"CSV 파일이 저장되었습니다: {os.path.abspath(csv_filename)}")
    return saved_data


if __name__ == "__main__":
    get_pose_angle()