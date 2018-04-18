# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:16:27 2017

@author: Jason
"""
#Import Libraries:
import vrep                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np         #array library
import math
import matplotlib as mpl   #used for image plotting
import time
import datetime
import random
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop


'''Establishes connection to the Simulation Server'''
vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
if clientID!=-1:  #check if client connection successful
    print ('Connected to remote API server')
    
else:
    print ('Connection not successful')
    sys.exit('Could not connect')
    
errorCodeLeftMotor,left_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_blocking)
errorCodeRightMotor,right_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)
vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)

vr = 1
vl = 1
outlier = 1.2
'''Gets the ids of the sensors of the robot'''
sH = []
s = []
for x in range(1,16+1):
    errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
    sH.append(sensor_handle)
s.append(sH[1]) 
s.append(sH[3]) 
s.append(sH[4])
s.append(sH[6])
s.append(sH[11])   



'''This is the neural network, it has a 0.1 droprate, a 5 sensor input and 5 array output
Relu function is used for activation and a rms optimizer.
We use a mean square error to track loss'''

model = Sequential()
#Layer 1
model.add(Dense(80, init='lecun_uniform', input_dim = 5, activation = 'relu'))
model.add(Dropout(0.1))
    
#Layer 2
model.add(Dense(40, init='lecun_uniform',activation = 'relu'))
model.add(Dropout(0.1))
    
model.add(Dense(3  , init='lecun_uniform',activation = 'linear'))
   
rms = RMSprop()
model.compile(loss = 'mse' , optimizer=rms, metrics=['accuracy'])



'''This is a function to return the sensor ids'''
def sensorHandles():
    sensor_h = []
    s = []
    for x in range(1,16+1):
        errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
        sensor_h.append(sensor_handle) #keep list of handles
    s.append(sH[1]) 
    s.append(sH[3]) 
    s.append(sH[4])
    s.append(sH[6])
    s.append(sH[11])    
    return s
    
    
'''This return the actual reading of the sensor, gibberish values are replaced with 1.2'''
def sensorInformation(sensor_handles, clientID):
    testInfo = []
    detectionInfo = []
    
    for x in range(5):
        errorCodeProximity,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handles[x-1],vrep.simx_opmode_streaming)                
     
    for x in range(5):
        errorCodeProximity,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handles[x-1],vrep.simx_opmode_buffer)   
        testInfo.append(np.linalg.norm(detectedPoint))
        detectionInfo.append(detectionState)

    sensorInformation = []
    for i in range(len(detectionInfo)):
        if detectionInfo[i]:
            sensorInformation.append(float(round(testInfo[i],3)))
        else:
            sensorInformation.append(outlier)

    
    return sensorInformation



'''This function moves the robot based on the prediction and returns the state after movement'''
def makeMove(state,action):
    factor = 0.8
        
    if action == 0:
        steerSpeed = factor/min(state[0],state[1])
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,steerSpeed, vrep.simx_opmode_streaming)  
        time.sleep(0.1)
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)  
        time.sleep(0.1)
        
    elif action == 1:
        steerSpeed = factor/min(state[2],state[3])
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,steerSpeed, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)  
        time.sleep(0.1)
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)  
        time.sleep(0.1)
        
    elif action == 2:
        steerSpeed = 0.12/vr
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,steerSpeed, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,steerSpeed, vrep.simx_opmode_streaming)
        
    sI = sensorInformation(s, clientID)
    return sI
    
'''The basic reward function used to train the network'''
def rewardFunction(state,action):


    if np.amin(state) < 0.11:
        return -15
    elif action == 0:
        return 0.5
    elif action == 1:
        return 0.5
    elif action == 2:
        return -0.1
    
    
def playing():
    c = 0
    model.load_weights('saved-models-modelWeight_buffer_3100.h5')
    sensor_h = sensorHandles()
    #Reading of all the current sensors
    while(1):
        state = sensorInformation(sensor_h, clientID)
        qVal = model.predict(np.array(state).reshape(1,5),batch_size = 1)
        action = np.argmax(qVal)
        new_state = makeMove(state,action)
        with open('Questions.txt', 'a') as writeFile:
            writeFile.write(str(state) + "," + str(action)+"\n")
        state = new_state
        
        if min(state) < 0.09:
            #sys.exit()
            vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
            time.sleep(1)
            vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
            c = c+1
            print(c)
            
playing()
    
    