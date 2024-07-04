import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('back_angles.csv')

# 평균 계산
average_angle = df['back_angle'].mean()

# 평균과 가장 가까운 값 찾기
closest_index = (df['back_angle'] - average_angle).abs().idxmin()
closest_time = df['back_time'][closest_index]
closest_angle = df['back_angle'][closest_index]

# 그래프 생성
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df['back_time'], df['back_angle'], marker='o')

# 평균과 가장 가까운 점 빨간색으로 표시
ax.plot(closest_time, closest_angle, 'ro', markersize=10)

# 그래프 제목과 축 레이블 설정
ax.set_title('Back Angle over Time', fontsize=16)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Back Angle (degrees)', fontsize=12)

# 격자 추가
ax.grid(True, linestyle='--', alpha=0.7)

# y축 범위 설정 (0도에서 최대각도까지)
ax.set_ylim(0, df['back_angle'].max() * 1.1)  # 최대값의 110%까지 표시

# 데이터 포인트에 값 표시
for i, txt in enumerate(df['back_angle']):
    ax.annotate(f'{txt:.1f}°', (df['back_time'][i], df['back_angle'][i]),
                 textcoords="offset points", xytext=(0,10), ha='center')

# 평균선 추가
ax.axhline(y=average_angle, color='r', linestyle='--')

# Average 값과 근사치를 그래프 오른쪽 위에 표시
ax.text(0.95, 0.95, f'Average: {average_angle:.2f}°\nClosest: {closest_angle:.2f}° at {closest_time:.2f}s',
        transform=ax.transAxes, fontsize=16, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# x축에 근사치 시간 표시
ax.annotate(f'{closest_time:.2f}s', (closest_time, 0), xytext=(0, -20),
            textcoords='offset points', ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

# 그래프 저장 및 표시
# plt.tight_layout()
# plt.savefig('back_angle_graph.png', bbox_inches='tight')
plt.show()