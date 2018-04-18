# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:31:59 2017

@author: Jason
"""

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
style.use('fivethirtyeight')



agent_plot = pd.read_csv("C:\\Users\\Jason\Desktop\\VREP Test multi agent - buffer\\epoch3_v2.txt",names =['ENo','Reward','Time','Epsilon'],sep = ',')
buffer_plot = pd.read_csv("C:\\Users\\Jason\\Desktop\\VREP Test multi agent - buffer\\epoch3_buffer.txt",names =['ENo','Reward','Time','Epsilon'],sep = ',')
agent_plot["Time"] = pd.to_datetime(agent_plot["Time"])
buffer_plot["Time"] = pd.to_datetime(buffer_plot["Time"])

#==============================================================================
# plt.rc('font', size=12)          # controls default text sizes
# plt.rc('axes', titlesize=12)     # fontsize of the axes title
# plt.rc('axes', labelsize= 12)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
# plt.rc('ytick', labelsize= 12)    # fontsize of the tick labels
# plt.rc('legend', fontsize= 12)    # legend fontsize
# plt.rc('figure', titlesize= 12)
# fig1 = plt.figure()
# fig1.patch.set_facecolor('white')
# plt.xlabel("No of Epochs")
# plt.ylabel("Rewards accumulated")
# 
# plt.plot(buffer_plot["ENo"],buffer_plot["Reward"],linewidth=1.0)
# plt.title("Epoch vs Reward with Buffer")
# 
# plt.show()
# 
# 
# 
# plt.rc('font', size=12)          # controls default text sizes
# plt.rc('axes', titlesize=12)     # fontsize of the axes title
# plt.rc('axes', labelsize= 12)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
# plt.rc('ytick', labelsize= 12)    # fontsize of the tick labels
# plt.rc('legend', fontsize= 12)    # legend fontsize
# plt.rc('figure', titlesize= 12)
# fig2 = plt.figure()
# fig2.patch.set_facecolor('white')
# plt.xlabel("No of Epochs")
# plt.ylabel("Epsilon")
# plt.plot(buffer_plot["ENo"],buffer_plot["Epsilon"],linewidth=1.0)
# plt.title("Epoch vs Epsilon with Buffer")
# 
# plt.show()
# 
# 
# 
# 
# 
# plt.rc('font', size=12)          # controls default text sizes
# plt.rc('axes', titlesize=12)     # fontsize of the axes title
# plt.rc('axes', labelsize= 12)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
# plt.rc('ytick', labelsize= 12)    # fontsize of the tick labels
# plt.rc('legend', fontsize= 12)    # legend fontsize
# plt.rc('figure', titlesize= 12)
# fig3 = plt.figure()
# fig3.patch.set_facecolor('white')
# plt.xlabel("No of Epochs")
# plt.ylabel("Rewards accumulated")
# plt.plot(agent_plot["ENo"],agent_plot["Reward"],linewidth=1.0)
# plt.title("Epoch vs Reward without Buffer")
# plt.show()
# 
# 
# 
# 
# 
# plt.rc('font', size=12)          # controls default text sizes
# plt.rc('axes', titlesize=12)     # fontsize of the axes title
# plt.rc('axes', labelsize= 12)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
# plt.rc('ytick', labelsize= 12)    # fontsize of the tick labels
# plt.rc('legend', fontsize= 12)    # legend fontsize
# plt.rc('figure', titlesize= 12)
# 
# fig4 = plt.figure()
# fig4.patch.set_facecolor('white')
# plt.xlabel("No of Epochs")
# plt.ylabel("Epsilon")
# 
# plt.plot(agent_plot["ENo"],agent_plot["Epsilon"],linewidth=1.0)
# plt.title("Epoch vs Epsilon without Buffer")
# 
# plt.show()
#==============================================================================



plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize= 12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize= 12)    # fontsize of the tick labels
plt.rc('legend', fontsize= 12)    # legend fontsize
plt.rc('figure', titlesize= 12)

fig5 = plt.figure()
fig5.patch.set_facecolor('white')
plt.xlabel("Time (hh:mm:ss)")
plt.ylabel("Epsilon")

plt.plot(agent_plot["Time"],agent_plot["Reward"],linewidth=1.0)
plt.title("Time vs Reward without Buffer")

plt.show()

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize= 12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize= 12)    # fontsize of the tick labels
plt.rc('legend', fontsize= 12)    # legend fontsize
plt.rc('figure', titlesize= 12)

fig5 = plt.figure()
fig5.patch.set_facecolor('white')
plt.xlabel("Time (hh:mm:ss)")
plt.ylabel("Epsilon")

plt.plot(buffer_plot["Time"],buffer_plot["Reward"],linewidth=1.0)
plt.title("Time vs Reward with Buffer")

plt.show()


