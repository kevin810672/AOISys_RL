# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 21:30:02 2018

@author: Chih Kai Cheng
"""
import os, datetime, cv2, glob, time, shutil, serial, sys



def multi_encoder(light):
    light0_hex = hex(light[0]).split('x')[-1].zfill(2).upper()
    light1_hex = hex(light[1]).split('x')[-1].zfill(2).upper()
    light2_hex = hex(light[2]).split('x')[-1].zfill(2).upper()
    light3_hex = hex(light[3]).split('x')[-1].zfill(2).upper()
    LRC = 255-(1+16+1+4+8+light[0]+light[1]+light[2]+light[3])+1
    while LRC<0:
        LRC += 256
    check = hex(LRC).split('x')[-1].zfill(2).upper()
    command = ":01100001000408%s%s%s%s%s\r\n" % (light0_hex, light1_hex, light2_hex, light3_hex, check)
    return command

def single_encoder(channel, light):
    light_hex = hex(light).split('x')[-1].zfill(4).upper()
    LRC = 255 - (1+6+channel+light)+1
    while LRC < 0:
        LRC += 255
    check = hex(LRC).split('x')[-1].zfill(2).upper()
    command = ":0106%s%s%s\r\n" % (str(channel).zfill(4), light_hex, check)
    return command

def action(expTime=None, light=None, channel=None):
    ser = serial.Serial(port="COM3",
                baudrate=19200,
                parity=serial.PARITY_NONE,
	                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS)   
    command = multi_encoder(light)
    ser.write(command.encode())
    ser.close()
    time.sleep(0.1)
    contents = [expTime, channel]
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Communication/"
    with open(path+"action.txt", "w+") as f:
        for item in contents:
            f.write("%s\n" % str(item))


def feedback(amount=1):
    path_pic = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Communication/pic/"
    imageList = glob.glob(path_pic+"*.jpg")
    while len(imageList) < amount:
        imageList = glob.glob(path_pic+"*.jpg")
    time.sleep(0.3)
    img = cv2.imread(imageList[0])
    while True:
        try:
            os.remove(imageList[0])
            break
        except(PermissionError):
            pass
    return img
            
def _buildExpFile(save=False):
    if save:  
        workSpace = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +"/Expirements/"
        if not os.path.exists(workSpace):
            os.mkdir(workSpace)
        DIR = workSpace + datetime.datetime.now().strftime("%Y%m%d")+'/'
        if not os.path.exists(DIR):
            os.mkdir(DIR)
        
        checkFile = os.listdir(DIR)
        if len(checkFile) == 0:
            os.mkdir(DIR + '0'.zfill(2))
            savePath = DIR + '0'.zfill(2) + '/'
        else:
            last = int(checkFile[-1])+1
            os.mkdir(DIR + str(last).zfill(2))
            savePath = DIR + str(last).zfill(2) + '/'
    else:
        savePath = "./"
    return savePath

def _closeLight():
    light = [0, 0, 0, 0]
    command = multi_encoder(light)
    ser = serial.Serial(port="COM3",baudrate=19200,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,
    	bytesize=serial.EIGHTBITS)
    ser.write(command.encode())
    ser.close()
       
def _cleanComFile():
    path_pic = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Communication/pic/"
    if os.path.exists(path_pic):
        shutil.rmtree(path_pic)
    while True:
        try:
            os.mkdir(path_pic)
            break
        except(PermissionError):
            pass
        
def timeit(func):
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', func.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ("%r: %3.2f ms" % \
                  (func.__name__, (te - ts) * 1000))
        return result
    return timed
    
def state(mode=0):
    if mode == 0:
        cv2.destroyAllWindows()
        _closeLight()
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Communication/"
    fp = open(path+"state.txt", "w+")
    while True:
        if not fp.closed:
            fp.write("%s\n" % str(mode))
            fp.close()
            break
        else:
            fp = open(path+"state.txt", "w+")

