import pandas as pd
import numpy as np
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
import time

#Load Road Data
road_data = np.genfromtxt('road_data.csv', delimiter = ',')

#Separte Target from Data
X = road_data[:,:3]
y = road_data[:,[3]]
y = y.ravel()
# Split cancerset into training (80%) and test (20%) set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)
# y_train_1 = y_train[y_train[:,0] == 1]
# y_train_0 = y_train[y_train[:,0] == 0]

# y_test_1 = y_test[y_test[:,0] == 1]
# y_test_0 = y_test[y_test[:,0] == 0]
# print(y_train.shape,y_train_1.shape,y_train_0.shape )
# print(y_test.shape,y_test_1.shape,y_test_0.shape)

# Use SVM
svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
# start = time.time()
clf = svc.fit(X_train, y_train)
stop = time.time()
print(f"Training time: {stop - start}s")
# Save trained model
filename = 'finalized_model.sav'
pickle.dump(svc, open(filename, 'wb'))

# Check Accuracy of Trained Model
svm_score = cross_val_score(svc, X, y, cv=10, scoring='accuracy', verbose = 1)
print("SVM accuracy: %0.2f (+/- %0.2f)" % (svm_score.mean(), svm_score.std() * 2))

# z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]
# tmp = np.linspace(-5,5,30)
# x,y = np.meshgrid(tmp,tmp)
# print(X_train)
# fig = plt.figure()
# ax  = fig.add_subplot(111, projection='3d')
# ax.set_title('Accelormeter Road Data')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# Good = ax.plot3D(X_train[y_train==0,0], X_train[y_train==0,1], X_train[y_train==0,2],'ob')
# Pothole = ax.plot3D(X_train[y_train==1,0], X_train[y_train==1,1], X_train[y_train==1,2],'sr')
# plt.legend([Good, Pothole], ["Good Road", "Pothole"])
# ax.plot_surface(x, y, z(x,y))
# ax.view_init(30, 60)
# plt.show()

# Test Train Model on Test Data
y_pred = svc.predict(X_test)

# Calculate Accuracry and the CM of Model 
num_correct = 0
for i in range(len(y_test)):
    if y_pred[i]==y_test[i]:
        num_correct +=1
        
Accuracy_rate = num_correct/len(y_test)
print("Accuracy Rate = ", Accuracy_rate)

cm = confusion_matrix(y_test, y_pred)
print(cm)


