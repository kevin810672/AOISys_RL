# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:23:16 2019

The algorithm is that using the reinforcement learning to autimatically tune the parameters of camera 
and lights in order to make the system intellent to find the best parameters such that the amount of 
defects can be maximum. 

@author: Chih Kai Cheng
"""

import func
from DDPG_discrete import DDPG, OUNoise, Discretization
from imageProcessing import imgReward
import numpy as np
import time
import cv2, sys



SAVEDIR = func._buildExpFile(save=True)

def aoil_record(value=None, item=None):
    if value is None:
        for item in item:
            f = open(SAVEDIR+item[0]+".txt", "w+")
            f.close()
    else:
        for val, item in zip(value, item):
            with open(SAVEDIR+item[0]+".txt", "a+") as f:
                f.seek(0)
                if not f.readlines():
                    f.write("%s\n" % str(0))
                f.write("%s\n" % str(val))

    
def aoil_imgShow(img, name="show", pos=None):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if not pos is None:
        cv2.moveWindow(name, pos[0], pos[1])
    cv2.imshow(name, img)
    cv2.waitKey(1)
    
def aoil_actSpace():
    actSpace = []
    actSpace_append = actSpace.append
    action_set   = 6
    action0_max  = 255
    action0_min  = 1
    action0_step = 1
    action1_max  = 19.1
    action1_min  = 0.1
    action1_step = 0.1
    for i in range(action_set):
        if not i%2:
            actSpace_append([action0_min, action0_max, action0_step])
        else:
            actSpace_append([action1_min, action1_max, action1_step])
    return actSpace

def aoil_step(action, times, maxStep=32):
    light = [0, 0, 0, 0]
    done_state = {0:"searching", 1:"max step", 2:"complete", 3:"worse"}
    channel = 0
    img_pool = []
    img_pool_append = img_pool.append
    for a in range(action.shape[0]):
        if not a%2:
            light[channel] = int(action[a])
        else:
            print("light{}| LI:{:03}\tET:{:05.2f} ms".format(a//2, light[channel], action[a]))
            func.action(expTime=action[a], light=light, channel=channel)
            img = func.feedback()
            # show the image
            aoil_imgShow(img)
            img_pool_append(img)
            light[channel] = 0
            channel += 1
    state_, reward, done = imgr.parse(img_pool, times)
    print("\nL1_D:{:.4e}\tL2_D:{:.4e}\tL3_D:{:.4e}\tLt_D:{:.4e}".format(state_[0], state_[2], state_[4], state_[6]))
    print("L1_N:{:.4e}\tL2_N:{:.4e}\tL3_N:{:.4e}\tLt_N:{:.4e}".format(state_[1], state_[3], state_[5], state_[7]))
    print("L1_R:{:.4e}\tL2_R:{:.4e}\tL3_R:{:.4e}\tLt_R:{:.4e}".format(reward[0], reward[2], reward[4], reward[6]))
    if times == maxStep:
        done[0] = 1
    print("\nState: {}".format(done_state[done[0]]))
    return state_, reward, done

def aoil_reset(s_dim):
    imgr.goalCounter = 0
    imgr.minCounter = 0
    imgr.maxReward = 0
    imgr.total_reward = 0
    imgr.maxpoint = 0
    ounoise.reset()
    return np.tile([0, 1], s_dim//2)

    
if __name__ == "__main__":
    actSpace = aoil_actSpace()
    A_DIM = len(actSpace)
    S_DIM = 8  # includes total defects, no.1 defects, no.1 noise...no3. defects, no3. noise
    
    # set the hyper parameters of DDPG
    LR_A, LR_C, GAMMA, TAU      = 1e-3, 1e-2, 9e-1, 1e-3
    MEMORY_CAPACITY, BATCH_SIZE = 800000, 100
    GRAPH, SAVE, LOAD           = False, True, False
    OU_MAX_SIGMA                = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])*5
    MAX_EPISODES                = 15000
    CROP_RATIO                  = 0.05
    NOISE_RATIO                 = 0.8
    MAX_STEP                    = 300
    ANG_TOLERANCE               = 3
    MASK_SIZE                   = 25
    DEF_AREA                    = 5
    DEF_QUANTITY                = 10000
    K_GROUP                     = [12, 9]
    # set the tracking item
    item_tracking = [["defLn0"], ["defLn1"], ["defLn2"],
                     ["Noise0"], ["Noise1"], ["Noise2"],
                     ["Light0_reward"], ["Light1_reward"], ["Light2_reward"],["Total_reward"],
                     ["Td_error"], ["A_loss"],
                     ["Light0"], ["ExpT0"], ["Light1"], ["ExpT1"], ["Light2"], ["ExpT2"]]
    
    # activate the state mornitor
    aoil_record(item=item_tracking)
    
    # initialize the DDPG model
    ddpg = DDPG(S_DIM, A_DIM, lr_a=LR_A, lr_c=LR_C, gamma=GAMMA, tau=TAU,\
                 memory_capacity=MEMORY_CAPACITY, batch_size=BATCH_SIZE,\
                 graph=GRAPH, save=SAVE)
    
    # initialize the OUNoise
    ounoise = OUNoise(actSpace=actSpace, max_sigma=OU_MAX_SIGMA, min_sigma=OU_MAX_SIGMA*0.1, decay_period = MAX_STEP)
    
    # initialize the Discretization of action
    discreteAct = Discretization(k=K_GROUP, actSpace=actSpace, method='euclidean')
    
    # initialize the imageprocess
    func.state(mode=1) # mode=1 : start the camera system
    imgr = imgReward(s_dim = S_DIM, a_dim = A_DIM,
                     crop_ratio  = CROP_RATIO,  ang_tol  = ANG_TOLERANCE, maskSize = MASK_SIZE,\
                     noise_ratio = NOISE_RATIO, def_area = DEF_AREA, lines = DEF_QUANTITY,\
                     savedir = SAVEDIR)
    
    # main loop
    t_start = time.time()
#    hist = {}
    for episode in range(MAX_EPISODES):
        r0, r2, r4, rt = [], [], [], []
        try:
            s = aoil_reset(S_DIM)
            times = 0
            while True:
                print("Ep:{:03}  Stp:{:03}".format(episode, times))
                t_step = time.time()
                a_r = ddpg.choose_action(s)
                a_o = ounoise.exploration(a_r, t=times)
                s_d, a_d = discreteAct.action(a_o, s)
                a = ddpg.evaluate_action(s_d, a_d)
                s_, r, done = aoil_step(a, times, maxStep=MAX_STEP)
                r0.append(r[0])
                r2.append(r[2])
                r4.append(r[4])
                rt.append(np.sum(imgr.reward[:-2])/2)
#                aoil_record(value = [r[0], r[2], r[4]], item  = [["Light1_reward"], ["Light2_reward"], ["Light3_reward"]])
                aoil_record(value = [a[0], a[1], a[2], a[3], a[4], a[5]], item=[["Light0"], ["ExpT0"], ["Light1"], ["ExpT1"], ["Light2"], ["ExpT2"]])
#                aoil_record(value = [r[0], r[2], r[4]], item  = [["Light0_reward"], ["Light1_reward"], ["Light2_reward"]])
                ddpg.store_transition(np.hstack((s, a, r[:-2], s_)))
                times += 1
                if ddpg.pointer > BATCH_SIZE:
                    ddpg.learn()
                    aoil_record(value=[ddpg.loss, ddpg.error], item=[["A_loss"], ["Td_error"]])        
                if times >= 200 :
                    func._closeLight()
#                    hist[imgr.maxReward] = imgr.maxpoint
                    ddpg.save(SAVEDIR+'model/', episode)
                    aoil_record(value = [sum(r0)/len(r0), sum(r2)/len(r2), sum(r4)/len(r4), sum(rt)/len(rt)], item = [["Light0_reward"], ["Light1_reward"], ["Light2_reward"], ["Total_reward"]])
                    print("Duration:{:.2f} s\n".format(time.time()-t_step))
                    print("%s\n" % "----------------------------------------------------------")
                    break
                s = s_
                print("Duration:{:.2f} s".format(time.time()-t_step))
                print("%s\n" % "----------------------------------------------------------")
        except(KeyboardInterrupt):
            cv2.destroyAllWindows()
            func._closeLight()
            sys.exit()
    # End: close light and destroy all windows
    t_end = time.time()
    print("Training Duration:{} hr {} min {} sec".format((t_end-t_start)//3600, (t_end-t_start)%3600//60, (t_end-t_start)%3600%60))
    func._closeLight()
    cv2.destroyAllWindows()
    func.state(mode=0)

    