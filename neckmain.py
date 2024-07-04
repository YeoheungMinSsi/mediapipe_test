from neckpose_angle_module import get_pose_angle

def main():
    saved_data = get_pose_angle(save_interval=10)  # 10초마다 각도 저장
    print("프로그램 종료 후 저장된 모든 데이터:")
    for data in saved_data:
        print(f"시간: {data[0]}초, 각도: {data[1]}도")

if __name__ == "__main__":
    main()
