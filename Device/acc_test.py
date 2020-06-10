import time
import board
import busio
import adafruit_adxl34x
 
i2c = busio.I2C(board.SCL, board.SDA)
accelerometer = adafruit_adxl34x.ADXL345(i2c)

accelerometer.enable_motion_detection()

while True:
    print("%f %f %f" % accelerometer.acceleration)

    print("Motion detected: %s" % accelerometer.events["motion"])
    time.sleep(0.5)
