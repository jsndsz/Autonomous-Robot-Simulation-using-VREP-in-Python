# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:56:48 2017

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
import random

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop


#Pre-Allocation

PI=math.pi  #pi=3.14..., constant
GAMMA = 0.9

vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)
modelHandle = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx',vrep.simx_opmode_blocking)
collisionHandle = vrep.simxGetCollisionHandle(clientID,'Pioneer_p3dx',vrep.simx_opmode_blocking)
cH = collisionHandle[0]
print(cH)


if clientID!=-1:  #check if client connection successful
    print ('Connected to remote API server')
    
else:
    print ('Connection not successful')
    sys.exit('Could not connect')


#retrieve motor  handles
errorCodeLeftMotor,left_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_blocking)
errorCodeRightMotor,right_motor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)
vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)
vl = 0.5
vr = 0.5


sensor_h=[] 
#sensor_loc=np.array([-PI/2, -50/180.0*PI,-30/180.0*PI,-10/180.0*PI,10/180.0*PI,30/180.0*PI,50/180.0*PI,PI/2,PI/2,130/180.0*PI,150/180.0*PI,170/180.0*PI,-170/180.0*PI,-150/180.0*PI,-130/180.0*PI,-PI/2]) 

#for loop to retrieve sensor arrays and initiate sensors
for x in range(1,16+1):
    errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
    sensor_h.append(sensor_handle) #keep list of handles

#dataset = numpy.loadtxt("NeuralNet.txt", delimiter=",")
model = Sequential()
#Layer 1
model.add(Dense(200, init='lecun_uniform', input_dim = 16, activation = 'relu'))
model.add(Dropout(0.1))
    
#Layer 2
model.add(Dense(150, init='lecun_uniform',activation = 'relu'))
model.add(Dropout(0.1))
    
model.add(Dense(5, init='lecun_uniform',activation = 'linear'))
   
rms = RMSprop()
model.compile(loss = 'mse' , optimizer=rms, metrics=['accuracy'])

#==============================================================================
# vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
# vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)
#==============================================================================



#0 - up
#1 - right
#2 - left
#3 - stop
#4 - reverse

def makeMove(state,action):
    
    if action == 0:
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl+1, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr+1, vrep.simx_opmode_streaming)
        
    elif action == 1:
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr+0.5, vrep.simx_opmode_streaming)    
        
    elif action == 2:
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl+0.5, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)
    
    elif action == 3:
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)
    
    elif action == 4:
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,-vl/2, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,-vr/2, vrep.simx_opmode_streaming)
    
    sensorInformation = np.array([])        
   
    for x in range(1,16+1):
        errorCodeProximity,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_h[x-1],vrep.simx_opmode_buffer)                
        sensorInformation=np.append(sensorInformation,np.linalg.norm(detectedPoint)) #get list of values
        #sensorInformation=np.append(sensorInformation,detectedPoint[2])
    k=""    
    with open('rawSensorInfo.txt', 'a') as the_file1:
        for i in range(len(sensorInformation)):
            k=k+","+str(sensorInformation[i])
        the_file1.write(k+"\n")
    
    return sensorInformation   


#0 - up
#1 - right
#2 - left
#3 - stop
#4 - reverse

def rewardFunction(state,action):
#==============================================================================
#     with open('stateFile.txt', 'a') as the_file:
#         k='current state:'+str(state)
#         the_file.write(k)
#==============================================================================

    if np.amin(state) < 0.15:
        return -15
    elif action == 0:
        return 3
    elif action == 1:
        return 1
    elif action == 2:
        return 1
    elif action == 3:
        return 0
    elif action == 4:
        return -1
    



def mini_batch_processing(mini_batch,model):
    Xtrain = []
    ytrain = []
    
    for m in mini_batch:
        old_state_m, action_m , reward_m , new_state_m = m
        old_state_m = old_state_m.reshape(1,16)
        new_state_m = new_state_m.reshape(1,16)
        
        
        old_qval = model.predict(old_state_m , batch_size = 1)
        new_qval = model.predict(new_state_m, batch_size = 1)

        maxQ = np.max(new_qval)
        y = np.zeros((1,5))
        y[:] = old_qval[:]
        if reward_m != -15:
            update = reward_m + (GAMMA * maxQ)
        else:
            update = reward_m
        
        y[0][action_m] = update
        Xtrain.append(old_state_m.reshape(1,16))
        ytrain.append(y.reshape(1,5))
    
    return np.array(Xtrain) , np.array(ytrain) 


yasas = 0
sensorInformation = np.array([])
for x in range(1,16+1):
    errorCodeSensor2,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
    sensor_h.append(sensor_handle) #keep list of handles
    errorCodeProximity2,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_h[x-1],vrep.simx_opmode_streaming)                
    sensorInformation=np.append(sensorInformation,np.linalg.norm(detectedPoint)) #get list of values
    #sensorInformation=np.append(sensorInformation,detectedPoint[2])
    yasas = yasas +1
    
    
def learning():
    
    
    frames = 100000
    observation = int(frames/100)
    instances = 5
    epsilon = 1
    batchSize = 60
    buffer = 500
    replay = []
    gamma = 0.9
    
    bufferCount = 0
    sensorInformation = np.array([])
    #capturing the current state
    returnCode,collisionState=vrep.simxReadCollision(clientID,cH,vrep.simx_opmode_streaming)


    for x in range(1,16+1):
        errorCodeProximity3,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_h[x-1],vrep.simx_opmode_streaming)                
        
    
    for x in range(1,16+1):
        errorCodeProximity3,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_h[x-1],vrep.simx_opmode_buffer)                
        sensorInformation=np.append(sensorInformation,np.linalg.norm(detectedPoint)) #get list of values
        #sensorInformation=np.append(sensorInformation,detectedPoint[2])
        print(detectedPoint)
        
    state=sensorInformation
         
    t = time.time()
    # no iter
    for i in range(frames):
        noOfCollisions=0
        #sensorInformation = np.array([])
        #check for randomness or for intially observation before training begins
        if (random.random() < epsilon) or (i < observation):
            action = random.randint(0,4)
        else:
            qVal = model.predict(state)
            action = np.argmax(qVal)
        
        new_state = makeMove(state,action)
        reward = rewardFunction(state,action)
        replay.append((state,action,reward,new_state))
        
        if i > observation:
            if len(replay) > buffer:
                replay.pop(0)
            
            #!!!!!check for batchSize, it may not be correct
            minibatch = random.sample(replay,batchSize)
            
            Xtrain , ytrain = mini_batch_processing(minibatch,model)
#==============================================================================
#             for j in range(batchSize):
#                 print("j is ",j)
#                 
#==============================================================================
 
            for i in range(batchSize):
                model.fit(Xtrain[i], ytrain[i], batch_size = 1, epochs = 1 , verbose = 1 )
        
        state = new_state
        
        if epsilon > 0.1 and i > observation:
            epsilon -= (1/frames)
        
        bufferCount+=1
        print("\n\n\n\n\nGame no: " ,bufferCount)
        print(state)
        
        k=""
        with open('stateFile.txt', 'a') as the_file:
            for i in range(len(state)):
                k=k+","+str(state[i])
            the_file.write(k+"\n")
            #write to file
        time.sleep(0.02)  
        
        returnCode,collisionState=vrep.simxReadCollision(clientID,cH,vrep.simx_opmode_buffer)
        print("Collison State: " , collisionState)
        #if np.amin(state) >= 0.7 and bufferCount>10:
        if collisionState==True:
            sys.exit("HERE: ")
            vrep.simxRemoveModel(clientID,modelHandle,vrep.simx_opmode_oneshot)
            vrep.simxLoadModel(clientID,modelHandle,0,vrep.simx_opmode_blocking)
        
        
#==============================================================================
#         
#         
#         #run an episode or game  for the given condition
#         while (noOfCollisions <1  and (time.time()-t) < 300):
#             qVal=model.predict(state.reshape(1,16),batch_size=batchSize)
#             action=(np.argmax(qVal))
#             #print("QVAL: " , qVal, "ACTION: ", action)
#             new_state=makeMove(state,action)
# 
#             reward=rewardFunction(new_state,action)
#             print("reward is ",reward)
#             if(reward==-15):
#                 noOfCollisions=noOfCollisions+1
#             print("no of cvol",noOfCollisions       )
#             #trial section
#            #==================================
# #==============================================================================
# #             X_train=[]
# #             y_train=[]
# #             y=np.zeros(5,1)
# #             y[:]=qVal[:]            
# #             X_train.append(state)
# #             y_train.append(y)
# #                 
# #             X_train=np.array(X_train)
# #             y_train=np.array(y_train)
# #             model.fit(X_train,y_train,batch_size=batchSize,nb_epoch=1,verbose=1)
# #             
# #==============================================================================
#             
#             #end here          
#             #=====================================
#             
#             
#             #store the state in the replay list
#             if(len(replay)<buffer):
#                 replay.append([state,action,reward,new_state])
#             else:
#                 if(bufferCount<buffer):
#                     bufferCount=bufferCount+1
#                 else:
#                     bufferCount=0
#                 replay[bufferCount] = [state,action,reward,new_state]
#                 
#             #next set training 
#             if len(replay) > buffer:
#                 miniBatch= random.sample(replay,batchSize)
#                 X_train=[]
#                 y_train=[]
#                 
#                 for mini in miniBatch:
#                     prev_state,action,reward,new_state=mini
#                     prev_qVal=model.predict(prev_state,batchSize)
#                     new_qVal=model.predict(new_state,batchSize)
#                     max_qVal=(np.amax(new_qVal))
#                     #y initialize to zero
#                     y=np.zeros(5,1)
#                     y[:]=prev_qVal[:]
#                     #terminating condition if you find any
#                     update=(reward+(gamma*max_qVal))
#                     #put the above update in the condition if u have any
#                     
#                     #not sure about this
#                     y[0][action]=update
#                     X_train.append(prev_state)
#                     y_train.append(y)
#                 
#                 X_train=np.array(X_train)
#                 y_train=np.array(y_train)
#                 # no idea about epoch and verbose here
#                 model.fit(X_train,y_train,batch_size=batchSize,nb_epoch=1,verbose=1)
#             
#             state = new_state
#         #reduce the epsilon
#         if epsilon> 0.1:#decide on this
#             epsilon-=(1/instances)
#==============================================================================
        
                
learning()               
                
                
            
            
            
        
    


