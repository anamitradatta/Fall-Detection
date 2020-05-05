import socket, sys
import numpy as np
import pickle
from features import extract_features
import threading
from time import sleep
isConnected=False
counter=0
prevAct=""
fallbool =False
startCount=False
class_names = ["falling", "jumping", "sitting", "standing","turning","walking"]  # ...

with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)

if classifier == None:
    print("Classifier is null; make sure you have trained it!")
    sys.exit()
    
def onActivityDetected(activity):
    global counter, fallbool, prevAct, startCount
    """
    Notifies the user of the current activity
    """
    
    if(activity=="falling" and prevAct!="falling" and (not startCount)):
        startCount=True
    
    if(startCount):
        counter+=1
    else:
        if(counter==0):
            counter=0
        else:
            counter-=1
    
    if(counter==4):
        print("Fall detected")
        counter=0
        startCount=False
    prevAct=activity

def predict(window):
    """
    Given a window of accelerometer data, predict the activity label.
    """

    # TODO: extract features over the window of data
    #print(window)
    feature_names, feature_vector = extract_features(window)

    # TODO: use classifier.predict(feature_vector) to predict the class label.

    # Make sure your feature vector is passed in the expected format
    class_label = classifier.predict(feature_vector.reshape(1, -1))

    # TODO: get the name of your predicted activity from 'class_names' using the returned label.
    # pass the activity name to onActivityDetected()

    activity_name = class_names[class_label.astype(int)[0]]

    onActivityDetected(activity_name)

    return

host = ''
port = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
s.bind((host, port))

sensor_data = []
window_size = 100  # ~1 sec assuming 25 Hz sampling rate
step_size = 100  # no overlap
index = 0  # to keep track of how many samples we have buffered so far

print("starting activity recognition for fall detection....")
sleep(10)
print("activity recognition started")
while 1:
    try:
        message, address = s.recvfrom(8192)
        info = message.decode()
        if(message and (not isConnected)):
            isConnected=True
            print("connected. receiving sensor data")
        #print (info)
        data = info.split(",")
        if(len(data)<13):
            continue
        
        #print(data)
        for i in range(9):
            
            data[i] = data[i].strip()
            #print(str(i) + " " + data[i])
        accel_x = data[2]
        accel_y = data[3]
        accel_z = data[4]
        gyro_x = data[6]
        gyro_y = data[7]
        gyro_z = data[8]
        tdata = np.asarray([accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z]).astype(np.float)
        
        sensor_data.append(tdata)
        index+=1
        
        while len(sensor_data) > window_size:
            sensor_data.pop(0)
        #print(tdata)
        
        
        if (index >= step_size and len(sensor_data) == window_size):
            t = threading.Thread(target=predict, args=(
            np.asarray(sensor_data[:]).astype(np.float),))
            t.start()
            index = 0
        
        sys.stdout.flush()
        
    except KeyboardInterrupt:
        print("interrupted")
