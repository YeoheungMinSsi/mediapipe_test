from neckpose_angle_module import get_pose_angle
import csv
import os
import numpy as np


def save_angles_to_csv(angles, filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['neck_time', 'neck_angle'])
        for time, angle in angles:
            writer.writerow([time, angle])
    print(f"CSV 파일이 저장되었습니다: {os.path.abspath(filename)}")


def create_angle_matrix(angles):
    return np.array(angles)


def main():
    saved_data = get_pose_angle(save_interval=2)  # 10초마다 각도 저장

    if saved_data:  # saved_data가 비어있지 않은 경우에만 처리
        print("기울기 각도 데이터:")
        for data in saved_data:
            print(f"시간: {data[0]}초, 각도: {data[1]}도")

        angle_matrix = create_angle_matrix(saved_data)
        print("\n기울기 각도 행렬:")
        print(angle_matrix)

        save_angles_to_csv(saved_data, 'neck_angles.csv')
    else:
        print("저장된 각도 데이터가 없습니다.")


if __name__ == "__main__":
    main()