# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:36:08 2019
@author: Chih Kai Cheng
Purpose: check out every single level of light and the exposure time of camera is prepared!
"""

import func
import numpy as np
import time, cv2, os, glob

def initialize():
    check = {"channel0":0, "channel1":0, "channel2":0}
    light = [0, 0, 0, 0]
    preGrayValue = 0
    NG = 0
    expMaxTime = []
    func._cleanComFile()
    t_start = time.time()
    t_step =[]
    for intensity in range(10, 40):
        for channel in range(3):
            light[channel] = intensity
            print(light)
            t = time.time()
            func.action(expTime=10, light=light, channel=channel)
            img = func.feedback(amount=1)
            func.imgShow(img, name="Hi")
            t_step.append(time.time()-t)
            img_mean_grayValue = np.mean(img)
            print("meanGray:{:.3f}\n".format(img_mean_grayValue))
            if (img_mean_grayValue-preGrayValue)<=0:
                NG += 1
                if NG > 2 :
                    check["channel"+str(channel)] = 1
            else:
                preGrayValue = img_mean_grayValue    
            light[channel] = 0
    t_light = time.time()-t_start
    for channel in range(0, 3):
        light[channel] = 255
        expT = 1
        while True:
            print(light)
            print(expT)
            
            func.action(expTime = expT, light=light, channel=channel)
            t1 = time.time()
            img = func.feedback(amount=1)
            t2 = time.time()
            print(t2-t1)
            img_mean_grayValue = np.mean(img)
            print("meanGray:{:.3f}\n".format(img_mean_grayValue))
            if img_mean_grayValue > 150:
                expMaxTime.append(expT)
                break
            elif expT > 10:
                expMaxTime.append(expT)
                break
            else:
                expT += 1
        light[channel] = 0
    t_exp = time.time()-t_start - t_light
    t_total = time.time()-t_start
    print("Total Time:{:.2f}s\nLight Time:{:.2f}s\nExp Time:{:.2f}s\n".format(t_total, t_light, t_exp))
    print("Average Time Step:{:2.2f}s\n".format(sum(t_step) / len(t_step)))
    return check, expMaxTime

func.state(mode=1)
check, expMaxTime = initialize()
#for c in range(0, 256):
#    print([c, 0, 0, 0])
#    func.action(expTime=5, light=[c, 0, 0, 0], channel=1)
#    img = func.feedback(amount=1)[0]
#    cv2.namedWindow("show", cv2.WINDOW_NORMAL)
#    cv2.imshow("show", img)
#    cv2.waitKey(1)
  
func.state(mode=0)