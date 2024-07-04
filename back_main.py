from backpose_angle_module import get_pose_angle
import csv
import os
import numpy as np


def save_angles_to_csv(angles, filename):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['back_time', 'back_angle'])
        for time, angle in zip(angles['back_time'], angles['back_angle']):
            writer.writerow([time, angle])
    print(f"CSV 파일이 저장되었습니다: {os.path.abspath(filename)}")


def create_angle_matrix(angles):
    return np.array([angles['back_time'], angles['back_angle']])


def main():
    angles = get_pose_angle(save_interval=2)  # 10초마다 각도 저장
    

    if angles:  # angles가 비어있지 않은 경우에만 처리
        print("기울기 각도 딕셔너리:")
        print("back_time:", angles['back_time'])
        print("back_angle:", angles['back_angle'])

        angle_matrix = create_angle_matrix(angles)
        print("기울기 각도 행렬:")
        print(angle_matrix)

        save_angles_to_csv(angles, 'back_angles.csv')
    else:
        print("저장된 각도 데이터가 없습니다.")


if __name__ == "__main__":
    main()