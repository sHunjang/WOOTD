import matplotlib.pyplot as plt

# Epoch 횟수
epochs = [10, 15, 20, 25, 30]

# 각 유형별 유사도
all_similar = [0.40, 0.61, 0.98, 0.98, 1.00]
bottom_similar = [0.40, 0.57, 0.94, 0.94, 0.99]
top_similar = [0.24, 0.36, 0.55, 0.55, 0.62]
all_different = [0.24, 0.34, 0.47, 0.47, 0.55]

# 각 유형별 그래프 그리기
plt.plot(epochs, all_similar, label='All Similar', marker='o')
plt.plot(epochs, bottom_similar, label='Bottom Similar', marker='o')
plt.plot(epochs, top_similar, label='Top Similar', marker='o')
plt.plot(epochs, all_different, label='All Different', marker='o')

# 그래프 제목과 축 레이블 설정
plt.title('Results')
plt.xlabel('Epochs')
plt.ylabel('Similarity')

# 범례 추가
plt.legend()

# 그래프 출력
plt.show()
