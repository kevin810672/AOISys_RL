# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:01:56 2019
@author: Chih Kai Cheng
Purpose: to visulize the learning process
"""
import matplotlib.pyplot as plt
from matplotlib import animation
import os, sys
from queue import Queue
from collections import OrderedDict

mode = {'Track':'0'}
input_mode = input("Track or Plot?(0/1): ")
# find the latest directory
desFilePath = os.path.join(os.path.dirname(os.getcwd()), "Expirements/")
if input_mode != mode['Track']:
    # analysis the input code
     if len(input_mode.split(" ")) >= 3:
        date = input_mode.split(" ")[1]
        number = input_mode.split(" ")[2]
        if len(input_mode.split(" ")) == 4:
            show_data_amount = int(input_mode.split(" ")[3])

        SAVEDIR = os.path.join(desFilePath, date, number)+"/"
     else:
        if len(input_mode.split(" ")) == 2:
            show_data_amount = int(input_mode.split(" ")[1])
        else:
            show_data_amount = None
        for _ in range(2):
            all_subdirs = [os.path.join(desFilePath, d) for d in os.listdir(desFilePath) if os.path.join(desFilePath, d)]
            SAVEDIR = max(all_subdirs, key=os.path.getmtime)
            desFilePath = SAVEDIR
        SAVEDIR = SAVEDIR+'\\'
else:
    for _ in range(2):
        all_subdirs = [os.path.join(desFilePath, d) for d in os.listdir(desFilePath) if os.path.join(desFilePath, d)]
        SAVEDIR = max(all_subdirs, key=os.path.getmtime)
        desFilePath = SAVEDIR
    SAVEDIR = SAVEDIR+'\\'


def aoil_StateTrack(tracking_item=None, figsize=(13, 9), layout=[2, 3]):
    item_num = len(tracking_item)
    fig = plt.figure(figsize=figsize, tight_layout=True)
    axs = [fig.add_subplot(layout[0], layout[1], i) for i in range(1, item_num+1)]       
    for ax, content in zip(axs, tracking_item):
        ax.set_title(content[0], fontsize = 10)
        ax.set_xlabel(content[1], fontsize = 10)
        ax.set_ylabel(content[2], fontsize = 10)
    plt.show()
    return fig, axs


def aoil_animate0(i, tracking_item, axs):
    show_data_amount = 200
    data_state = q0.get()
    for index, (content, ax) in enumerate(zip(tracking_item, axs)):
        y = open(SAVEDIR+content[0]+".txt", "r+").readlines()
        if not y:
            y = ['0\n']
        else:
            y = [float(value.split('\n')[0]) for value in y]
            x = list(range(0, len(y)))
            # only show the last 1000 data
            if len(y) > show_data_amount:
                ax.set_xlim(len(y)-show_data_amount, len(y))
                y = y[-show_data_amount:]
                x = x[-show_data_amount:]
            if not (index//3)%2 or index==9:
                maxR = round(max(y), 3)
                if maxR > data_state[index]:
                    data_state[index] = maxR
                else:
                    maxR = data_state[index]
                ax.plot(x, y, lw=0.5, color='r')
                ax.plot([], [], color='white', label="Max: "+str(maxR))
                handles,labels=ax.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                ax.legend([list(by_label.values())[0]], [list(by_label.keys())[0]],handlelength=0, loc='best')
            else:
                minL = round(min(y[1:]), 3)
                if minL < data_state[index]:
                    data_state[index] = minL
                else:
                    minL = data_state[index]
                ax.plot(x, y, lw=0.5, color='b')
                ax.plot([], [], color='white', label="Min: "+str(minL))
                handles,labels=ax.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                ax.legend([list(by_label.values())[-1]], [list(by_label.keys())[-1]],handlelength=0, loc='best')
    q0.put(data_state)
    
def aoil_animate1(i, tracking_item, axs):
    show_data_amount = 200
    data_state = q1.get()
    for index, (content, ax) in enumerate(zip(tracking_item, axs)):
        y = open(SAVEDIR+content[0]+".txt", "r+").readlines()
        if not y:
            y = ['0\n']
        else:
            y = [float(value.split('\n')[0]) for value in y]
            x = list(range(0, len(y)))
            # only show the last 1000 data
            if len(y) > show_data_amount:
                ax.set_xlim(len(y)-show_data_amount, len(y))
                y = y[-show_data_amount:]
                x = x[-show_data_amount:]
        
            ax.plot(x, y, lw=0.5, color='g')
            if index<3:
                ax.plot([], [], color='white', label="lvl_curr:"+str(y[-1]))
            else:
                ax.plot([], [], color='white', label="Exp_curr:"+str(y[-1])+" ms")
            handles, labels=ax.get_legend_handles_labels()
            ax.legend([handles[-1]], [labels[-1]], handlelength=0, loc='best')
    q1.put(data_state)
    
def aoil_dataPlt0(tracking_item, axs, amount=None):
    data_state = q0.get()
    for index, (content, ax) in enumerate(zip(tracking_item, axs)):
        try:
            y = open(SAVEDIR+content[0]+".txt", "r+").readlines()
        except:
            print("The file does not exist!")
            sys.exit()
        if not y:
            y = ['0\n']
        else:
            y = [float(value.split('\n')[0]) for value in y]
            x = list(range(0, len(y)))
            y = y[:show_data_amount]
            x = x[:show_data_amount]
            if not (index//3)%2 or index==9:
                maxR = round(max(y), 3)
                if maxR > data_state[index]:
                    data_state[index] = maxR
                else:
                    maxR = data_state[index]
                ax.plot(x, y, lw=0.5, color='r')
                ax.plot([], [], color='white', label="Max: "+str(maxR))
                handles,labels=ax.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                ax.legend([list(by_label.values())[0]], [list(by_label.keys())[0]],handlelength=0, loc='best')
            else:
                minL = round(min(y[1:]), 3)
                if minL < data_state[index]:
                    data_state[index] = minL
                else:
                    minL = data_state[index]
                ax.plot(x, y, lw=0.5, color='b')
                ax.plot([], [], color='white', label="Min: "+str(minL))
                handles,labels=ax.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                ax.legend([list(by_label.values())[-1]], [list(by_label.keys())[-1]],handlelength=0, loc='best')

def aoil_dataPlt1(tracking_item, axs, amount=None):
    for index, (content, ax) in enumerate(zip(tracking_item, axs)):
        try:
            y = open(SAVEDIR+content[0]+".txt", "r+").readlines()
        except:
            print("The file does not exist!")
            sys.exit()
        if not y:
            y = ['0\n']
        else:
            y = [float(value.split('\n')[0]) for value in y]
            x = list(range(0, len(y)))
            y = y[:show_data_amount]
            x = x[:show_data_amount]
            ax.plot(x, y, lw=0.5, color='g')
            ax.plot([], [], color='white', label="Max:"+str(max(y[1:]))+"\nMin:"+str(min(y[1:])))
            ax.legend(handlelength=0, loc='best')

            
            
if __name__ == "__main__":
    item0 = [["defLn0", "steps", "pixels"], ["defLn1", "steps", "pixels"], ["defLn2", "steps", "pixels"],
             ["Noise0", "steps", "ratio"], ["Noise1", "steps", "ratio"], ["Noise2", "steps", "ratio"],
             ["Light0_reward", "episode", "scores"], ["Light1_reward", "episode", "scores"], ["Light2_reward", "episode", "scores"],
             ["Total_reward", "episode", "scores"],["Td_error", "steps", "values"], ["A_loss", "steps", "values"]]
#    item0 = [["defLn0", "steps", "pixels"], ["defLn1", "steps", "pixels"], ["defLn2", "steps", "pixels"],
#             ["Noise0", "steps", "ratio"], ["Noise1", "steps", "ratio"], ["Noise2", "steps", "ratio"],
#             ["Light0_reward", "episode", "scores"], ["Light1_reward", "episode", "scores"], ["Light2_reward", "episode", "scores"],
#             ["Td_error", "steps", "values"], ["A_loss", "steps", "values"]]
    item1 = [["Light0", "steps", "lvl"], ["Light1", "steps", "lvl"], ["Light2", "steps", "lvl"], 
             ["ExpT0", "steps", "ms"], ["ExpT1", "steps", "ms"], ["ExpT2", "steps", "ms"],]
    q0 = Queue()
    q0.put([0 for _ in range(len(item0))])
    q1 = Queue()
    q1.put([0 for _ in range(len(item1))])
    fig0, axs0 = aoil_StateTrack(tracking_item = item0, figsize=(13,9), layout=[4, 3])
    fig1, axs1 = aoil_StateTrack(tracking_item = item1, figsize=(13,5), layout=[2, 3])
    
    
    if input_mode == mode['Track']:
        ani0 = animation.FuncAnimation(fig0, aoil_animate0, fargs=(item0, axs0), interval=1000, blit=False)
        ani1 = animation.FuncAnimation(fig1, aoil_animate1, fargs=(item1, axs1), interval=1000, blit=False)
    else:
        aoil_dataPlt0(item0, axs0, amount=show_data_amount)
        aoil_dataPlt1(item1, axs1, amount=show_data_amount)
        