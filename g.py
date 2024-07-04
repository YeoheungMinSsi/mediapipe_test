import matplotlib.pyplot as plt
import numpy as np

# 거북목 증후군 진료 인원 추이 데이터
years = [2011, 2012, 2013, 2014, 2015]
patients = [606, 726, 846, 1004, 1134]

# 연령별 인구 비율 데이터
age_groups = ['10s', '20s', '30s']
population_per_100k = [196, 268, 234]

# 그래프 설정
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 거북목 증후군 진료 인원 추이 물결 그래프
ax1.plot(years, patients, marker='o', linestyle='-', color='b')
ax1.set_title('Trend of Patients with Text Neck Syndrome')
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Patients')
ax1.set_ylim([0, 2000])  # y축 범위 설정
ax1.set_yticks(np.arange(0, 2001, 200))  # 눈금선 0부터 2000까지, 간격 200

# 연령별 인구 비율 그래프
ax2.bar(age_groups, population_per_100k)
ax2.set_title('Population per 100,000 by Age Group')
ax2.set_xlabel('Age Group')
ax2.set_ylabel('Population per 100,000')

# 간격 조정
plt.subplots_adjust(wspace=0.5)

# 그래프 출력
plt.show()
