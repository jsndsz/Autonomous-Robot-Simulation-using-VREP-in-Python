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
    
errorCodeLeftMotor,left_motor_handle1=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_blocking)
errorCodeRightMotor,right_motor_handle1=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)
vrep.simxSetJointTargetVelocity(clientID,left_motor_handle1,0, vrep.simx_opmode_streaming)
vrep.simxSetJointTargetVelocity(clientID,right_motor_handle1,0, vrep.simx_opmode_streaming)

errorCodeLeftMotor,left_motor_handle2=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor#0',vrep.simx_opmode_blocking)
errorCodeRightMotor,right_motor_handle2=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor#0',vrep.simx_opmode_oneshot_wait)
vrep.simxSetJointTargetVelocity(clientID,left_motor_handle2,0, vrep.simx_opmode_streaming)
vrep.simxSetJointTargetVelocity(clientID,right_motor_handle2,0, vrep.simx_opmode_streaming)

errorCodeLeftMotor,left_motor_handle3=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor#1',vrep.simx_opmode_blocking)
errorCodeRightMotor,right_motor_handle3=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor#1',vrep.simx_opmode_oneshot_wait)
vrep.simxSetJointTargetVelocity(clientID,left_motor_handle3,0, vrep.simx_opmode_streaming)
vrep.simxSetJointTargetVelocity(clientID,right_motor_handle3,0, vrep.simx_opmode_streaming)

errorCodeLeftMotor,left_motor_handle4=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor#2',vrep.simx_opmode_blocking)
errorCodeRightMotor,right_motor_handle4=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor#2',vrep.simx_opmode_oneshot_wait)
vrep.simxSetJointTargetVelocity(clientID,left_motor_handle4,0, vrep.simx_opmode_streaming)
vrep.simxSetJointTargetVelocity(clientID,right_motor_handle4,0, vrep.simx_opmode_streaming)



vr = 1
vl = 1
outlier = 1.2
'''Gets the ids of the sensors of the robot'''
sH1 = []
s1 = []
for x in range(1,16+1):
    errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
    sH1.append(sensor_handle)
s1.append(sH1[1]) 
s1.append(sH1[3]) 
s1.append(sH1[4])
s1.append(sH1[6])
s1.append(sH1[11])   


sH2 = []
s2 = []
for x in range(1,16+1):
    errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x)+'#0',vrep.simx_opmode_oneshot_wait)
    sH2.append(sensor_handle)
s2.append(sH2[1]) 
s2.append(sH2[3]) 
s2.append(sH2[4])
s2.append(sH2[6])
s2.append(sH2[11])   


sH3 = []
s3 = []
for x in range(1,16+1):
    errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x)+'#1',vrep.simx_opmode_oneshot_wait)
    sH3.append(sensor_handle)
s3.append(sH3[1]) 
s3.append(sH3[3]) 
s3.append(sH3[4])
s3.append(sH3[6])
s3.append(sH3[11])   

sH4 = []
s4 = []
for x in range(1,16+1):
    errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x)+'#2',vrep.simx_opmode_oneshot_wait)
    sH4.append(sensor_handle)
s4.append(sH4[1]) 
s4.append(sH4[3]) 
s4.append(sH4[4])
s4.append(sH4[6])
s4.append(sH4[11])   



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
	sH1 = []
	s1 = []
	for x in range(1,16+1):
		errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
		sH1.append(sensor_handle)
	s1.append(sH1[1]) 
	s1.append(sH1[3]) 
	s1.append(sH1[4])
	s1.append(sH1[6])
	s1.append(sH1[11])   


	sH2 = []
	s2 = []
	for x in range(1,16+1):
		errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x)+'#0',vrep.simx_opmode_oneshot_wait)
		sH2.append(sensor_handle)
	s2.append(sH2[1]) 
	s2.append(sH2[3]) 
	s2.append(sH2[4])
	s2.append(sH2[6])
	s2.append(sH2[11])   


	sH3 = []
	s3 = []
	for x in range(1,16+1):
		errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x)+'#1',vrep.simx_opmode_oneshot_wait)
		sH3.append(sensor_handle)
	s3.append(sH3[1]) 
	s3.append(sH3[3]) 
	s3.append(sH3[4])
	s3.append(sH3[6])
	s3.append(sH3[11])   

	sH4 = []
	s4 = []
	for x in range(1,16+1):
		errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x)+'#2',vrep.simx_opmode_oneshot_wait)
		sH4.append(sensor_handle)
	s4.append(sH4[1]) 
	s4.append(sH4[3]) 
	s4.append(sH4[4])
	s4.append(sH4[6])
	s4.append(sH4[11])   
	
	return s1,s2,s3,s4
    
    
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
def makeMove(state,action,sensorinfo,lmh,rmh):
    factor = 0.8
        
    if action == 0:
        steerSpeed = factor/min(state[0],state[1],lmh,rmh)
        vrep.simxSetJointTargetVelocity(clientID,lmh,0, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,rmh,steerSpeed, vrep.simx_opmode_streaming)  
        time.sleep(0.2)
        vrep.simxSetJointTargetVelocity(clientID,lmh,vl, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,rmh,vr, vrep.simx_opmode_streaming)  
        time.sleep(0.1)
        
    elif action == 1:
        steerSpeed = factor/min(state[2],state[3])
        vrep.simxSetJointTargetVelocity(clientID,lmh,steerSpeed, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,rmh,0, vrep.simx_opmode_streaming)  
        time.sleep(0.2)
        vrep.simxSetJointTargetVelocity(clientID,lmh,vl, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,rmh,vr, vrep.simx_opmode_streaming)  
        time.sleep(0.1)
        
    elif action == 2:
        steerSpeed = 0.12/vr
        vrep.simxSetJointTargetVelocity(clientID,lmh,steerSpeed, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,rmh,steerSpeed, vrep.simx_opmode_streaming)
        time.sleep(0.2)
        
    sI = sensorInformation(sensorinfo, clientID)
    return sI

    
def playing():
    c = 0
    model.load_weights('saved-models-modelWeight_buffer_3100.h5')
    sensor_h1,sensor_h2,sensor_h3,sensor_h4 = sensorHandles()
    print(sensor_h1,sensor_h2,sensor_h3,sensor_h4)
    #Reading of all the current sensors
    while(1):
        state1 = sensorInformation(sensor_h1, clientID)
        qVal1 = model.predict(np.array(state1).reshape(1,5),batch_size = 1)
        action1 = np.argmax(qVal1)
        new_state1 = makeMove(state1,action1,sensor_h1,left_motor_handle1,right_motor_handle1)
        state1 = new_state1
        
		
        state2 = sensorInformation(sensor_h2, clientID)
        qVal2 = model.predict(np.array(state2).reshape(1,5),batch_size = 1)
        action2 = np.argmax(qVal2)
        new_state2 = makeMove(state2,action2,sensor_h2,left_motor_handle2,right_motor_handle2)
        state2 = new_state2
		
		
        state3 = sensorInformation(sensor_h3, clientID)
        qVal3 = model.predict(np.array(state3).reshape(1,5),batch_size = 1)
        action3 = np.argmax(qVal3)
        new_state3 = makeMove(state3,action3,sensor_h3,left_motor_handle3,right_motor_handle3)
        state3 = new_state3
		
		
        state4 = sensorInformation(sensor_h4, clientID)
        qVal4 = model.predict(np.array(state4).reshape(1,5),batch_size = 1)
        action4 = np.argmax(qVal4)
        new_state4 = makeMove(state4,action4,sensor_h4,left_motor_handle4,right_motor_handle4)
        state4 = new_state4
		
		
		
        if min(state1) < 0.09 or min(state2) < 0.09 or min(state3) < 0.09 or min(state4) < 0.09:
            #sys.exit()
            vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
            time.sleep(1)
            vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
            c+=1
            print(c)
            
playing()
    
    