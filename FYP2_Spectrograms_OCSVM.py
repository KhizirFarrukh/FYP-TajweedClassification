import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score

image_directory = os.path.join(os.getcwd(),'TrainingQ_spectrograms')

X = []
for filename in os.listdir(image_directory):
  if filename.endswith('.png'):
    image_path = os.path.join(image_directory, filename)
    image = plt.imread(image_path)
    X.append(image)

X = np.array(X)

X = X.reshape(X.shape[0], -1)

model = OneClassSVM(kernel='poly', nu=0.1)
model.fit(X)

y_pred_train = model.predict(X)

accuracy = accuracy_score(np.ones_like(y_pred_train), y_pred_train)
print("Accuracy:", accuracy)
