import serial
import numpy as np
#import pandas as pd
import pickle
from sklearn import svm
filename = 'finalized_model.sav'


f = open("CommTest.txt", "a")

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

    ser.flush()

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            #print(line)
            if(line != "Calibrate" and len(line.split()) > 8):
                f.write(line)
                f.write("\n")
#Gets data into np.array
                newData = line.split()
                #print(newData)
                #print(newData[2])
                #print(newData[5])
                #print(newData[8])
                Data = [newData[2], newData[5], newData[8]]
                #print(Data)
                Data = np.array([float(Data[0]), float(Data[1]), float(Data[2])])
                #print(Data)
                #print(Data.dtype)
                #print(Data[0].dtype)
#Now to get the ML stuff
                loaded_model = pickle.load(open(filename, 'rb'))
                print(Data)
                Data = Data.reshape(1, -1)
                y_pred = loaded_model.predict(Data)
                print(y_pred)
