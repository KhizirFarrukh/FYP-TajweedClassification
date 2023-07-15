import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Accuracy_Scores_Test_Qalqalah.csv')

grouped_data = data.groupby('kernel_mode')

for kernel_mode, group in grouped_data:
    plt.plot(group['nu_value'], group['accuracy'], label=kernel_mode)

plt.xlabel('nu_value')
plt.xlim(0.04, 0.21)

plt.ylabel('accuracy')
plt.ylim(0, 1)

plt.legend()

plt.show()
