from backpose_angle_module import get_pose_angle
import csv
import os
import numpy as np


def save_angles_to_csv(angles, filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['neck_time', 'neck_angle'])
        for time, angle in zip(angles['neck_time'], angles['neck_angle']):
            writer.writerow([time, angle])
    print(f"CSV 파일이 저장되었습니다: {os.path.abspath(filename)}")


def create_angle_matrix(angles):
    return np.array([angles['neck_time'], angles['neck_angle']])


def main():
    angles = get_pose_angle()
    print("기울기 각도 딕셔너리:")
    print("neck_time:", angles['neck_time'])
    print("neck_angle:", angles['neck_angle'])

    angle_matrix = create_angle_matrix(angles)
    print("기울기 각도 행렬:")
    print(angle_matrix)

    save_angles_to_csv(angles, 'angles.csv')


if __name__ == "__main__":
    main()