import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

road_data = np.genfromtxt('road_data.csv', delimiter = ',')

X = road_data[:,:3]
y = road_data[:,[3]]

# Split cancerset into training (80%) and test (20%) set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)

# svc = svm.SVC(kernel='linear', C=1)
# svc.fit(X_train, y_train)

filename = 'finalized_model.sav'
# pickle.dump(svc, open(filename, 'wb'))

# svm_score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
# print("SVM accuracy: %0.2f (+/- %0.2f)" % (svm_score.mean(), svm_score.std() * 2))

loaded_model = pickle.load(open(filename, 'rb'))

y_pred = loaded_model.predict(X_test)

num_correct = 0
for i in range(len(y_test)):
    if y_pred[i]==y_test[i]:
        num_correct +=1
        
Accuracy_rate = num_correct/len(y_test)
print("Accuracy Rate = ", Accuracy_rate)

cm = confusion_matrix(y_test, y_pred)
print(cm)


