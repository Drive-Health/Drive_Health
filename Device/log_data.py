import os
import threading
import urllib.request


def sendDataToServer():

	threading.Timer(600,sendDataToServer).start()
	print("Sensing...")
	Pothole = True
	Lo = 10.25
	La = 40.56
	print(Pothole)
	print(Lo)
	print(La)
	Lo ="%.1f" %Lo
	La = "%.1f" %La
	urllib.request.urlopen("http://drivehealth.000webhostapp.com/api/insert.php?longitude="+Lo+"&latitude="+La).read()

sendDataToServer()
