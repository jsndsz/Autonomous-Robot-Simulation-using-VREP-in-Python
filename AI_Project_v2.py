# -- coding: utf-8 --
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


frames = 1000
observation = 20
epsilon = 1
batchSize = 200
buffer = 20000
GAMMA = 0.9
vl = 2 #left motor velocity
vr = 2 #right motor velocity
outlier = 1.2
alpha = 1
PI = 3.14
sensor_loc=np.array([-PI/2, -50/180.0*PI,-30/180.0*PI,-10/180.0*PI,10/180.0*PI,30/180.0*PI,50/180.0*PI,PI/2,PI/2,130/180.0*PI,150/180.0*PI,170/180.0*PI,-170/180.0*PI,-150/180.0*PI,-130/180.0*PI,-PI/2]) 
softening_factor = 0.3

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


'''Gets the ids of the sensors of the robot'''
sH = []
s = []
sl= []
for x in range(1,16+1):
    errorCodeSensor,sensor_handle=vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor'+str(x),vrep.simx_opmode_oneshot_wait)
    sH.append(sensor_handle)
s.append(sH[1]) 
s.append(sH[3]) 
s.append(sH[4])
s.append(sH[6])
s.append(sH[11])   
sl.append(sensor_loc[1]) 
sl.append(sensor_loc[3]) 
sl.append(sensor_loc[4])
sl.append(sensor_loc[6])
sl.append(sensor_loc[11])  


'''This is the neural network, it has a 0.5 droprate, a 5 sensor input and 5 array output
Relu function is used for activation and a rms optimizer.
We use a mean square error to track loss'''

model = Sequential()
#Layer 1
model.add(Dense(80, init='lecun_uniform', input_dim = 5, activation = 'relu'))
model.add(Dropout(0.5))
    
#Layer 2
model.add(Dense(40, init='lecun_uniform',activation = 'relu'))
model.add(Dropout(0.5))
    
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
        #time.sleep(0.007)
        #sensorInformation=np.append(sensorInformation,np.linalg.norm(detectedPoint)) #get list of values
       
    for x in range(5):
        errorCodeProximity,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=vrep.simxReadProximitySensor(clientID,sensor_handles[x-1],vrep.simx_opmode_buffer)   
        testInfo.append(np.linalg.norm(detectedPoint))
        detectionInfo.append(detectionState)
        #detectionInfo[sensor_h[x-1]] = detectionState
        #time.sleep(0.007)

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
    
#==============================================================================
#     if action == 0:
#         vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl+1, vrep.simx_opmode_streaming)
#         vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr+1, vrep.simx_opmode_streaming)
# 
#==============================================================================
    
    if action == 0:
        steerSpeed = factor/min(state[0],state[1])
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,steerSpeed, vrep.simx_opmode_streaming)  
        time.sleep(0.2)
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)  
        time.sleep(0.1)
#==============================================================================
#         time.sleep(0.01)
#         vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl-1, vrep.simx_opmode_streaming)
#         vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr-1, vrep.simx_opmode_streaming)
#         time.sleep(0.01)
#==============================================================================
        
    elif action == 1:
        steerSpeed = factor/min(state[2],state[3])
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,steerSpeed, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)  
        time.sleep(0.2)
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)  
        time.sleep(0.1)
#==============================================================================
#         time.sleep(0.01)
#         vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl-1, vrep.simx_opmode_streaming)
#         vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr-1, vrep.simx_opmode_streaming)
#         time.sleep(0.01)
#==============================================================================
        
    elif action == 2:
        steerSpeed = 0.12/vr
        vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,steerSpeed, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,steerSpeed, vrep.simx_opmode_streaming)
#==============================================================================
#         time.sleep(0.01)
#         vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
#         vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)
#         time.sleep(0.01)
#==============================================================================
#==============================================================================
#     elif action == 4:
#         vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,-vl/2, vrep.simx_opmode_streaming)
#         vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,-vr/2, vrep.simx_opmode_streaming)
#         vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
#         vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)
#     
#==============================================================================
    sI = sensorInformation(s, clientID)
    return sI
    
'''The basic reward function used to train the network'''
def rewardFunction(state,action):
#==============================================================================
#     with open('stateFile.txt', 'a') as the_file:
#         k='current state:'+str(state)
#         the_file.write(k)
#==============================================================================

    if np.amin(state) < 0.11:
        return -5
    elif action == 0:
        return 0.5
    elif action == 1:
        return 0.5
    elif action == 2:
        return -0.1
#==============================================================================
#     elif action == 3:
#         return 0
#==============================================================================
#==============================================================================
#     elif action == 4:
#         return -1
#     
#==============================================================================
'''This function takes a random sample from a buffer of 'state,action,reward,new_state' pairs and processes it to 
establish X_train and y_train sets.
A basic Q Learning update formula is used and the element which needs to be influences is replaced in the actual y and returned'''
def mini_batch_processing(mini_batch):
    Xtrain = []
    ytrain = []
    
    for m in mini_batch:
        old_state_m, action_m , reward_m , new_state_m = m
        old_state_m = np.array(old_state_m).reshape(1,5)
        new_state_m = np.array(new_state_m).reshape(1,5)
        
        
        old_qval = model.predict(old_state_m , batch_size = 1)
        new_qval = model.predict(new_state_m, batch_size = 1)

        maxQ = np.max(new_qval)
        y = np.zeros((1,3))
        y[:] = old_qval[:]
#==============================================================================
#         for i in range(len(y)):
#             y[i] = (y[i] - np.mean(y))/np.std(y)
#==============================================================================
        if reward_m != -5:
            update = (1-alpha)*reward_m +  alpha*(reward_m + (GAMMA * maxQ))
        else:
            update = reward_m
        
        y[0][action_m] = update
        Xtrain.append(old_state_m.reshape(1,5))
        ytrain.append(y.reshape(1,3))
    
    return Xtrain , ytrain


'''This function is where all the action happens'''
def learning():
    #create a buffer to store the state action reward and new_state
    replay = []
    agent_number = 1
    xaxis = []
    yaxis = []
    tr=0
    #This factor decides the probability of random action or predicted action. It is decreased lineraly over time
    epsilon = 1 
    #ID s of the sensors
    sensor_h = sensorHandles()
    #Reading of all the current sensors
    state = sensorInformation(sensor_h, clientID)
   
    #Keeps a track of the total reward 
    total_reward = 0
    i = 1
    bf = 0
    #Each frame is from start state to collision
    while i < frames:
        #noOfCollisions=0
        bf+=1 
        
        #check for randomness or for intially observation before training begins
        if (random.random() < epsilon) or (i < observation):
            action = random.randint(0,2)
            print("RANDOMMM")
            print("EPISILON: " ,epsilon)
        else:
            print("CRASHHHHHH")
            print("EPISILON: " ,epsilon)
            #This is the prediction from the network, not certain if this is how you give input, but we do recieve an output
            qVal = model.predict(np.array(state).reshape(1,5),batch_size = 1)
            action = np.argmax(qVal)
        
        #Builiding up the buffer
        new_state = makeMove(state,action)
        reward = rewardFunction(state,action)
        replay.append((state,action,reward,new_state))
        total_reward+=reward
        
        #Keeps a check on the length of the buffer and always rmeoves the first element and appends below FIFO
        if i > observation:
            if len(replay) > buffer:
                replay.pop(0)
            
            #!!!!!check for batchSize, it may not be correct
            #Produces the random batch samples
            minibatch = random.sample(replay,batchSize)
            
            
            #Gets back the Xtrain and ytrain
            Xtrain , ytrain = mini_batch_processing(minibatch)


#==============================================================================
# with open('stateFile.txt', 'a') as the_file:
#             for i in range(len(state)):
#                 k=k+","+str(state[i])
#             the_file.write(k+"\n")
#==============================================================================
            

            Training_X= np.array(Xtrain).reshape(batchSize,5)
            Training_Labels = np.array(ytrain).reshape(batchSize,3)
            time.sleep(0.01)
            model.fit(Training_X, Training_Labels, batch_size = batchSize , epochs = 1 , verbose = 1 )
            time.sleep(0.01)
            #Writing the weights to file
            
            
            if i % 10 == 0:
                
                model.save_weights('saved-models-' + 'modelWeight_agent_'+str(agent_number) + str(i) + '.h5',overwrite=True)
                z='buffer'+ str(agent_number) + str(i)+".txt"
                with open(z, 'w') as writeFile:
                    writeFile.write(str(replay))

                
#             with open('TrainingSet.txt', 'a') as the_file:
#                 for i in range(batchSize):
#                     model.fit(Xtrain[i], ytrain[i], batch_size = 1, nb_epoch = 1 , verbose = 1 )
#                     for z in Xtrain[i]:
#                         trainsamp=trainsamp+str(z)+","
#                     #xstr=Xtrain[i]
#                     xstr=trainsamp
#                     ystr=ytrain[i]
#                     m=m+"\nXtrain is for level :"+str(i)+"is: "+str(xstr)+" \nytrain is:"+str(ystr)+"\n\n\n"
#                     the_file.write(m)
#                 #model.fit(Xtrain, ytrain, batch_size = batchSize, nb_epoch = 1 , verbose = 1 )
#                     #model.save_weights('saved-models-' + 'modelWeight' +'.h5',overwrite=True)
#==============================================================================
                
        #Decresaing the value of epsilon over time
        state = new_state
        print(state)
        if epsilon > 0.1 and i > observation:
            epsilon -= (1/1000)
        
        print("\n\n Game no : " , i)
        
#==============================================================================
#         k=""
#         with open('stateFile.txt', 'a') as the_file:
#             for j in range(len(state)):
#                 k=k+","+str(state[j])
#             the_file.write(k+"\n")
#==============================================================================
        #Write to file to later analyse result
        if min(state) < 0.09:
            t = datetime.datetime.now().time()
            with open('epoch1.txt', 'a') as writeFile:
                writeFile.write(str(i) + "," + str(total_reward) + "," + str(t)+","+str(epsilon)+"\n")
            xaxis.append(i)
            yaxis.append(total_reward)
            i = i+1
            print(total_reward)
            total_reward = 0
            print("Buffer: " , bf)
            #sys.exit()
            vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
            time.sleep(1)
            vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
            
        if i%100 == 0 :
            plt.plot(xaxis,yaxis)
            plt.show()


learning()
