# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 01:48:48 2017

@author: syd24
"""

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
import threading
import os.path
import ast
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop


agent_number=3
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

agent_number=3


vrep.simxFinish(-1) # just in case, close all opened connections
'''Establishes connection to the Simulation Server'''

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


class agent1():
    
    def __init__(self):
        self.agent_number=3
        self.neighbouring_buffer=[]
        self.model = Sequential()
        #Layer 1
        self.model.add(Dense(80, init='lecun_uniform', input_dim = 5, activation = 'relu'))
        self.model.add(Dropout(0.1))
        #Layer 2
        self.model.add(Dense(40, init='lecun_uniform',activation = 'relu'))
        self.model.add(Dropout(0.1))
        #Layer 3    
        self.model.add(Dense(3  , init='lecun_uniform',activation = 'linear'))
           
        rms = RMSprop()
        self.model.compile(loss = 'mse' , optimizer=rms, metrics=['accuracy'])

    #Method to frame the questions and answers
    def QuestionSet(self):
        noOfQuestions=10
        q1=q2=q3=[]
        questions=[]
        
        
        q1=[1.2, 1.2, 0.072, 1.2, 1.2]
        q2=[1.2, 0.373, 1.2, 1.2, 1.2]
        q3=[1.2, 0.365, 1.2, 1.2, 1.2]
        q4=[1.2, 0.389, 1.2, 1.2, 1.2]
        q5=[1.2, 0.277, 0.117, 1.2, 1.2]
        q6=[1.2, 0.263, 0.079, 1.2, 1.2]
        q7=[1.2, 0.264, 0.081, 1.2, 1.2]
        q8=[1.2, 1.2, 0.071, 1.2, 1.2]
        q9=[1.2, 0.249, 0.071, 1.2, 1.2]
        q10=[1.2, 0.729, 1.2, 1.2, 0.526]
        
        questions=[q1,q2,q3,q4,q5,q6,q7,q8,q9,q10]
        answers=[2,1,1,1,2,2,2,2,2,0]
        
            
        return questions,answers
    
    def actionSteps(self,fitlist):
        flist=[]
        na=np.array(fitlist)
        for i in range(len(na)):
            c=np.argmax(na[i])
            flist.append(c)
        return flist
        
    
    #Method to return top two best trained agents
    def rankAgents(self,fscore):
        totalScore=[]
        questionSet=[]
        questionSet,answerSet=self.QuestionSet()
    
        for i in range(len(fscore)):
            reward=0
            for j in fscore[i]:
                if j==answerSet[j]:
                    reward=reward+1;
            totalScore.append(reward)
        
        return totalScore
    
    
    
    def checkModel(self,counter_file):
        finalScore=[]
        indexOfTopTwo=[]
        merged_buffer=[]
        
        x2="saved-models-modelWeight_agent_1"+str(counter_file)+".h5"
        x3="saved-models-modelWeight_agent_2"+str(counter_file)+".h5"
        x4="saved-models-modelWeight_agent_4"+str(counter_file)+".h5"
        buffer_file2='buffer1'+str(counter_file)+'.txt'
        buffer_file3='buffer2'+str(counter_file)+'.txt'
        buffer_file4='buffer4'+str(counter_file)+'.txt'

        questions,answers=self.QuestionSet()
    
        if(os.path.exists(x2) and os.path.exists(x3) and os.path.exists(x4) and os.path.exists(buffer_file2) and os.path.exists(buffer_file3) and os.path.exists(buffer_file4)  ):
            print("value of i",counter_file)
            
            ######## Model Formation===========================

            #===========================Model Formation ends here
            
            
            ########## Loading the models and performing a prediction
            self.model.load_weights(x2)
            yFit2 = self.model.predict(questions, batch_size=1)
            
            self.model.load_weights(x3)
            yFit3 = self.model.predict(questions, batch_size=1)
            
            self.model.load_weights(x4)
            yFit4 = self.model.predict(questions, batch_size=1)
            #=============================Prediction ends
            
            
            ####### Formulation  finalscore==============
            final_Fit2=self.actionSteps(yFit2)
            final_Fit3=self.actionSteps(yFit3)
            final_Fit4=self.actionSteps(yFit4)
            fscore=[final_Fit2,final_Fit3,final_Fit4]
            
            finalScore=self.rankAgents(fscore)
            #======================== ends
            
            #Fetches the index of top two elements
            indexOfTopTwo=sorted(range(len(finalScore)), key=lambda i: finalScore[i])[-2:]
            
            ####### Fetching all the agents buffers ==============
                    
            buffer2 = []
            buffer3 = []
            buffer4= []
                        
            ############Buffer Filled in===============
            f2 = open(buffer_file2, "r")
            f3 = open(buffer_file3, "r")
            f4 = open(buffer_file4, "r")
            
            buffer2=f2.read()
            buffer3=f3.read()
            buffer4=f4.read()
            
            buffer2 = ast.literal_eval(buffer2)
            buffer3 = ast.literal_eval(buffer3)
            buffer4 = ast.literal_eval(buffer4)
            #==============================ends
            
            
            mid_buffer=[buffer2,buffer3,buffer4]
            print("\n\n\n\nIndex: " ,indexOfTopTwo)
            ############## Merging the top 2 buffers=============
            for i in indexOfTopTwo:
               merged_buffer.append(mid_buffer[i])
                    
            final_merged_buffer=[]
            
            for x in range(2):
                for z in merged_buffer[x]:
                    final_merged_buffer.append(z)
            #==============================ends
      
            
            self.set_neighbouring_buffer(final_merged_buffer)
            
            ################ Update file names to check for the next model set
#==============================================================================
#             x2="saved-models-modelWeight_2"+str(counter_file+10)+".h5"
#             x3="saved-models-modelWeight_3"+str(counter_file+10)+".h5"
#             x4="saved-models-modelWeight_4"+str(counter_file+10)+".h5"
#             buffer_file2='buffer2'+str(counter_file+10)+'.txt'
#             buffer_file3='buffer3'+str(counter_file+10)+'.txt'
#             buffer_file4='buffer4'+str(counter_file+10)+'.txt'
#==============================================================================
            #==========================================ends
           
            
        else:
            print("not there")
            #sleeep for a min 
            
            
            
    def set_neighbouring_buffer(self,buffer):
        self.neighbouring_buffer=buffer
    
    
    def sensorHandles(self):
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
    def sensorInformation(self,sensor_handles, clientID):
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
    def makeMove(self,state,action):
        factor = 0.8
            
        if action == 0:
            steerSpeed = factor/min(state[0],state[1])
            vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,0, vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,steerSpeed, vrep.simx_opmode_streaming)  
            time.sleep(0.2)
            vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)  
            time.sleep(0.1)
            
        elif action == 1:
            steerSpeed = factor/min(state[2],state[3])
            vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,steerSpeed, vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,0, vrep.simx_opmode_streaming)  
            time.sleep(0.2)
            vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,vl, vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,vr, vrep.simx_opmode_streaming)  
            time.sleep(0.1)
            
        elif action == 2:
            steerSpeed = 0.12/vr
            vrep.simxSetJointTargetVelocity(clientID,left_motor_handle,steerSpeed, vrep.simx_opmode_streaming)
            vrep.simxSetJointTargetVelocity(clientID,right_motor_handle,steerSpeed, vrep.simx_opmode_streaming)
            
        sI = self.sensorInformation(s, clientID)
        return sI
        
    '''The basic reward function used to train the network'''
    def rewardFunction(self,state,action):
    
    
        if np.amin(state) < 0.11:
            return -5
        elif action == 0:
            return 0.5
        elif action == 1:
            return 0.5
        elif action == 2:
            return -0.1
    
    '''This function takes a random sample from a buffer of 'state,action,reward,new_state' pairs and processes it to 
    establish X_train and y_train sets.
    A basic Q Learning update formula is used and the element which needs to be influences is replaced in the actual y and returned'''
    def mini_batch_processing(self,mini_batch,model1):
        Xtrain = []
        ytrain = []
        
        for m in mini_batch:
            old_state_m, action_m , reward_m , new_state_m = m
            old_state_m = np.array(old_state_m).reshape(1,5)
            new_state_m = np.array(new_state_m).reshape(1,5)
            
            
            old_qval = model1.predict(old_state_m , batch_size = 1)
            new_qval = model1.predict(new_state_m, batch_size = 1)
    
            maxQ = np.max(new_qval)
            y = np.zeros((1,3))
            y[:] = old_qval[:]
    
            if reward_m != -5:
                update = (1-alpha)*reward_m +  alpha*(reward_m + (GAMMA * maxQ))
            else:
                update = reward_m
            
            y[0][action_m] = update
            Xtrain.append(old_state_m.reshape(1,5))
            ytrain.append(y.reshape(1,3))
        
        return Xtrain , ytrain
    
    
    '''This function is where all the action happens'''
    def learning(self):
        # counter files
                
        model1 = Sequential()
        #Layer 1
        model1.add(Dense(80, init='lecun_uniform', input_dim = 5, activation = 'relu'))
        model1.add(Dropout(0.1))
            
        #Layer 2
        model1.add(Dense(40, init='lecun_uniform',activation = 'relu'))
        model1.add(Dropout(0.1))
            
        model1.add(Dense(3  , init='lecun_uniform',activation = 'linear'))
           
        rms = RMSprop()
        model1.compile(loss = 'mse' , optimizer=rms, metrics=['accuracy'])


        #create a buffer to store the state action reward and new_state
        replay = []
        xaxis = []
        yaxis = []
        #This factor decides the probability of random action or predicted action. It is decreased lineraly over time
        epsilon = 1 
        #ID s of the sensors
        sensor_h = self.sensorHandles()
        #Reading of all the current sensors
        state = self.sensorInformation(sensor_h, clientID)
       
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
                qVal = model1.predict(np.array(state).reshape(1,5),batch_size = 1)
                action = np.argmax(qVal)
            
            #Builiding up the buffer
            new_state = self.makeMove(state,action)
            reward = self.rewardFunction(state,action)
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
                Xtrain , ytrain = self.mini_batch_processing(minibatch,model1)
    
                #This fits the Xtrain and ytrain to the model
                #It may be problematic sending it in batches of such size in this manner.
                #This started working only after trail and error, so can be one of the potential issues
                #The training accuracy remains constant
                Training_X= np.array(Xtrain).reshape(batchSize,5)
                Training_Labels = np.array(ytrain).reshape(batchSize,3)
                time.sleep(0.01)
                model1.fit(Training_X, Training_Labels, batch_size = batchSize , epochs = 1 , verbose = 1 )
                time.sleep(0.01)
                
                ############ Training on neighboring buffers=========================
                if(self.neighbouring_buffer):
                    print("\n\n\n\n\nBUFFEERRRR EXCHANNNGEEEE")

                    minibatch_exchange = random.sample(self.neighbouring_buffer,25)
                    #Gets back the Xtrain and ytrain
                    Xtrain_exchange , ytrain_exchange = self.mini_batch_processing(minibatch_exchange,model1)
                    #This fits the Xtrain and ytrain to the model
                    #It may be problematic sending it in batches of such size in this manner.
                    #This started working only after trail and error, so can be one of the potential issues
                    #The training accuracy remains constant
                    Training_X_exchange= np.array(Xtrain_exchange).reshape(25,5)
                    Training_Labels_exchange = np.array(ytrain_exchange).reshape(25,3)
                    time.sleep(0.01)
                    model1.fit(Training_X_exchange, Training_Labels_exchange, batch_size = 25 , epochs = 1 , verbose = 1 )
                    self.neighbouring_buffer=[]

                   
                #======================================= Buffer Exchange
                
                #Writing the weights to file
                if i % 10 == 0:
                    
                    self.checkModel(i)
                    model1.save_weights('saved-models-' + 'modelWeight_buffer_'+str(agent_number) + str(i) + '.h5',overwrite=True)
# =============================================================================
#                     tr=i%10
#                     
# =============================================================================
                    
                        
            #Decresaing the value of epsilon over time
            state = new_state
            if epsilon > 0.1 and i > observation:
                epsilon -= (1/1000)
            
            print("\n\n Game number----------------- : " , i)
            
            #Write to file to later analyse result
            if min(state) < 0.09:
                t = datetime.datetime.now().time()
                with open('epoch3_buffer.txt', 'a') as writeFile:
                    writeFile.write(str(i) + "," + str(total_reward) + "," + str(t)+","+str(epsilon)+"\n")
                xaxis.append(i)
                yaxis.append(total_reward)
                i = i+1
                total_reward = 0
                #sys.exit()
                vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
                time.sleep(1)
                vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot)
                

obj=agent1()
obj.learning()