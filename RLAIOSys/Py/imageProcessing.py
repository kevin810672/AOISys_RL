# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:29:57 2019

-image processing
    using simple image processing technique to extract the defects and noise
    from three different light image.
    Nosie metric reference: No-Reference Image Quality Assessment using Blur and Noise
    binary method: just only extract the relative high frequency pixels
@author: Chih Kai Cheng
"""

import cv2, glob, os, math, time, sys
import numpy as np
import func


class imgCalibration(object):
    def __init__(self, tolerance=None, crop_ratio=None, maskSize=None):
        self.tolerance=tolerance
        self.rangeRatio = 0.15
        self.lineThresh , self.minLineLen, self.maxLineGap= 5, 20, 5
        self.anchorGroupThresh = 60
        self.crop_ratio, self.maskSize = crop_ratio, maskSize
        self.tempL, self.tempW, self.templateCircleCenterX, self.templateCircleCenterY = 895, 915, 8.95, 9.3
        self.templateCircleBRadius, self.templateCircleSRadius = 6.8, 3.5
        self.img_maskCreate()

    def calibrate(self, img):
        print("Anchor detection and image calibration...")
        img_process = self.img_fft(img)
        line_h, line_v = self.img_anchorPos(img_process)
        pos_line, rotate_angle = self.img_intersect(line_h, line_v)
        # image show test
# =============================================================================
#         test = img.copy()
#         for x1,y1,x2,y2 in pos_line[:]:      
#             cv2.line(test,(x1,y1),(x2,y2),(0,255,0),5)
#         self.imgshow(test, obs=True)
# =============================================================================
        print("Angle of anchor point: {:.2f} degree".format(rotate_angle))
        self.rotate_angle = rotate_angle
        if abs(rotate_angle) < self.tolerance:
            print("Angle of anchor point: in the tolerance("+str(self.tolerance)+" degree)\n")
        else:
            img= self.img_rotate(img, -rotate_angle)
            img_process = self.img_fft(img)
            line_h, line_v = self.img_anchorPos(img_process)
            
            pos_line, rotate_angle = self.img_intersect(line_h, line_v)
            print("Calibration of anchor point:{:.2f} degree\n".format(rotate_angle))

        self.calib_points = [pos_line[0,1], pos_line[0,1]+self.tempW, pos_line[0,0], pos_line[0,0]+self.tempL]
  
    def img_preprocess(self, img, dilateTimes=None, erodeTimes=None, edge=False):
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_binary = cv2.threshold(img_gray, np.mean(img_gray)+1*np.std(img_gray), 255, cv2.THRESH_BINARY)[1]
        img_binary = cv2.medianBlur(img_binary, 7)
        element_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        element_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
        img_dilate = cv2.dilate(img_binary, element_dilate, iterations=dilateTimes)
        img_erode = cv2.erode(img_dilate, element_erode, iterations=erodeTimes)
        if edge:
            img_edges = cv2.Canny(img_erode, 0, 255)
            return img_edges
        else:
            return img_erode
        
    def img_fft(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # fft
        dft = np.fft.fft2(img_gray)
        # fft shift
        dft_shift = np.fft.fftshift(dft)
        # filter out the puxels to contain which we want (cross pattern)
        bandwidth = 6
        dft_shift[:int(img.shape[0]/2)-bandwidth, :int(img.shape[1]/2)-bandwidth] = 0
        dft_shift[:int(img.shape[0]/2)-bandwidth, int(img.shape[1]/2)+bandwidth:] = 0
        dft_shift[int(img.shape[0]/2)+bandwidth:, :int(img.shape[1]/2)-bandwidth] = 0
        dft_shift[int(img.shape[0]/2)+bandwidth:, int(img.shape[1]/2)+bandwidth:] = 0
        # inverse fft
        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = np.uint8(np.abs(np.fft.ifft2(f_ishift)))
  
        # binarize and using canny edge detector to catch the edge
        img_binary = cv2.threshold(img_back, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        img_edges = cv2.Canny(img_binary, 0, 255)
        return img_edges
    
    def img_rotate(self, img, angle):
        h, w = img.shape[:2]
        cX, cY = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return cv2.warpAffine(img, M, (nW, nH))

    def img_anchorPos(self, img):
        # set the line threshold to check whether the edges we choose are the true lines
        lines = cv2.HoughLinesP(img,1.,np.pi/180,self.lineThresh,minLineLength=self.minLineLen,maxLineGap=self.maxLineGap)[:,0,:]
        # find the line around the position pattern, the ratio sets 0.2
        line = []
        line_append = line.append
        for x1,y1,x2,y2 in lines[:]:
            if x2<img.shape[1]*self.rangeRatio*2 and y2<img.shape[0]*self.rangeRatio:
                line_append([x1,y1,x2,y2])

        # seperate the different slop lines into two group inlcuding -45 to 45 degrees and the other
        line_h = []
        line_h_append = line_h.append
        line_v = []
        line_v_append = line_v.append
        for pos in line:
            degree = math.degrees(math.atan((pos[3]-pos[1])/(pos[2]-pos[0]+1e-6)))
            if degree <= self.anchorGroupThresh and degree >= -self.anchorGroupThresh:
                line_h_append(pos)
            else:
                line_v_append(pos)
        return np.asarray(line_h), np.asarray(line_v)

    def img_intersect(self, line_h, line_v):
        # find the intersection of line1 and line2
        # first, find the two line function
        line , a, b = [], [], []
        line.append(np.mean(line_h, axis=0))
        line.append(np.mean(line_v, axis=0))
        
        for l in line:
            a.append((l[3] - l[1]) / (l[2] - l[0] + 1e-6))
            b.append(l[1] - (l[3] - l[1]) / (l[2] - l[0] + 1e-6) * l[0])
            
        intersect_x = int((b[0] - b[1]) / ( a[1] - a[0] + 1e-6))
        intersect_y = int(a[0] * (b[0] - b[1]) / ( a[1] - a[0] + 1e-6) + b[0])
        pos_line = np.asarray([[intersect_x, intersect_y, int(line[0][2]), int(line[0][3])], [intersect_x, intersect_y, int(line[-1][2]), int(line[-1][3])]])
        rotate_angle = math.atan((line[0][3]-intersect_y) / (line[0][2]-intersect_x+1e-6))
        return pos_line, math.degrees(rotate_angle)

    def img_maskCreate(self):
        read_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/Templates/Image/"
        pic_name = glob.glob(read_path+"*.jpg")
        if not pic_name:
            print("There is not any image file in /Templates/Image!")
            sys.exit()
        else:
            img = cv2.imread(pic_name[0])
            self.calibrate(img)
            img_pos = img[self.calib_points[0]:self.calib_points[1], self.calib_points[2]:self.calib_points[3]]
            # mask create
            circles = []
            img_mask = np.zeros_like(img_pos)
            # circle position and the radius
            c0_center_x, c0_center_y, c0_radius = img_pos.shape[1]//2-35, img_pos.shape[0]//2+43, 180 
            cl_center_x, cl_center_y, cl_radius = img_pos.shape[1]//2-320, img_pos.shape[0]//2+43, 10
            cr_center_x, cr_center_y, cr_radius = img_pos.shape[1]//2+305, img_pos.shape[0]//2+18, 10
            cu_center_x, cu_center_y, cu_radius = img_pos.shape[1]//2-20, img_pos.shape[0]//2-278, 10
            cd_center_x, cd_center_y, cd_radius = img_pos.shape[1]//2+5, img_pos.shape[0]//2+350, 10
            clu_center_x, clu_center_y, clu_radius = img_pos.shape[1]//2-270, img_pos.shape[0]//2-190, 50
            cru_center_x, cru_center_y, cru_radius = img_pos.shape[1]//2+180, img_pos.shape[0]//2-190, 50
            cld_center_x, cld_center_y, cld_radius = img_pos.shape[1]//2-270, img_pos.shape[0]//2+260, 50
            crd_center_x, crd_center_y, crd_radius = img_pos.shape[1]//2+180, img_pos.shape[0]//2+260, 50
            circles.append([c0_center_x, c0_center_y, c0_radius])
            circles.append([cl_center_x, cl_center_y, cl_radius])
            circles.append([cr_center_x, cr_center_y, cr_radius])
            circles.append([cu_center_x, cu_center_y, cu_radius])
            circles.append([cd_center_x, cd_center_y, cd_radius])
            circles.append([clu_center_x, clu_center_y, clu_radius])
            circles.append([cru_center_x, cru_center_y, cru_radius])
            circles.append([cld_center_x, cld_center_y, cld_radius])
            circles.append([crd_center_x, crd_center_y, crd_radius])
            
            
            for index, pts in enumerate(circles):
                if index == 0:
                    # draw the outer circle with white color ## using img_mask->img_pos:to check whether you design the wrong mask 
                    img_mask = cv2.circle(img_mask,(pts[0],pts[1]),pts[2],(255,255,255), self.maskSize)
                else:
                    img_mask = cv2.circle(img_mask,(pts[0],pts[1]),pts[2],(255,255,255), -1)

            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            self.img_mask = img_mask[int(img_mask.shape[0]*self.crop_ratio):int(img_mask.shape[0]*(1-self.crop_ratio)),\
                                     int(img_mask.shape[1]*self.crop_ratio):int(img_mask.shape[1]*(1-self.crop_ratio))]
            self.imgshow(self.img_mask, obs=True)

    def imgshow(self, img, name="show", obs=False, pos=None):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if not pos is None:
            cv2.moveWindow(name, pos[0], pos[1])
        cv2.imshow(name, img)
        if not obs:
            cv2.waitKey(1)
        else:
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class imgReward(object):
    def __init__(self, s_dim       = None, a_dim     = None,\
                       crop_ratio  = 0.08, ang_tol   = 1,  maskSize = 5,\
                       noise_ratio = 0.5,  def_area = 10, lines    = 100,\
                       savedir = None):
        
        self.s_dim , self.a_dim = s_dim, a_dim
        self.done = np.zeros((1), dtype=np.uint8)
        self.reward = np.zeros((s_dim), dtype=np.float32)
        self.state = np.zeros((s_dim), dtype=np.float32)
        self.criteria = np.zeros((a_dim), dtype=np.float32)
        self.epison = 1e-6
        self.crop_ratio = crop_ratio
        self.w0, self.w1 = (1-noise_ratio)/200, noise_ratio
        self.def_area , self.lines = def_area, lines
        self.savedir = savedir if savedir is not None else "./temp/"
        self.recData = []
        self.recCounter = {}
        self.imgc = imgCalibration(tolerance=ang_tol, crop_ratio=crop_ratio, maskSize=maskSize)
        print("Mask generation.....done")
        print("Image precessing....done\n")
#        self.imgshow(self.imgc.img_mask, obs=True)
        self.minCounter= 0
        self.goalCounter = 0
        self.maxReward = -1
        self.total_reward = 0
        self.maxpoint = 0
        
    def parse(self, image_pool, times, rec=True):
        self.step = times
        self.image = image_pool
        self.imgCalib()
        self.detDefLn(record=rec)
        self.detNoise(record=rec)
        
        # intergate two image metrics to be state
        self.state = np.vstack((np.asarray(self.defect_ratio), np.asarray(self.noise_metric))).T.reshape(-1)
        self.rewFunc()    
        return self.state, self.reward, self.done
    
            
    def imgCalib(self):
        for index in range(len(self.image)):
            if abs(self.imgc.rotate_angle) > self.imgc.tolerance:
                self.image[index] = self.imgc.img_rotate(self.image[index], -self.imgc.rotate_angle)
            # extract the images that are in the aclibration figures
            self.image[index] = self.image[index][self.imgc.calib_points[0]:self.imgc.calib_points[1],\
                                                  self.imgc.calib_points[2]:self.imgc.calib_points[3]]
            # crop the images to eliminate calibration figures
            self.image[index] = self.image[index][int(self.image[index].shape[0]*self.crop_ratio):int(self.image[index].shape[0]*(1-self.crop_ratio)),\
                                                  int(self.image[index].shape[1]*self.crop_ratio):int(self.image[index].shape[1]*(1-self.crop_ratio))]
            # convert the images into gray-lavel images
            self.image[index] = cv2.cvtColor(self.image[index], cv2.COLOR_BGR2GRAY)
            
    def detDefLn(self,record=True):
        self.defect_ratio = []
        self.D = []
        defect_ratio_append = self.defect_ratio.append
        overall = []
        overall_append = overall.append
        for index, img in enumerate(self.image):            
            # denoise
            img = cv2.GaussianBlur(img, (3, 3), 0)
            if index == 2:
                # inverse the image of outer-axial light 0->255 
                img = np.uint8(255 - img)
            # utilize the sobel edge to detect the edges: it's more useful than the canny detecor
            x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            edge = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            edge[np.where(edge>(np.mean(edge) + 3*np.std(edge)))] = 255
            edges = cv2.medianBlur(edge, 5)
            
            # in order to constraint the much bright and much dark images: -> failed
            if np.sum(edge==255) > img.size*0.15:
                edges = np.zeros_like(img)
            # collect the edge image
            overall_append(edges)
#            self.imgshow(edge, "defLn_nm")
        # collect the overall image
        overall_append(overall[0] | overall[1] | overall[2])
        for index, edge in enumerate(overall):
            edge_mask = edge & self.imgc.img_mask
            if index==3:
                 self.imgshow(edge_mask, "AL_defLn_m")
            else:
                self.imgshow(edge_mask, "defLn_m")
            line_def = self.BLOB(edge_mask)
            defect_ratio_append(line_def)
            if index==3:
                self.D.append(line_def)
            if record and index < 3:
                self.record(data={'format':'.jpg', 'type':'defLn', 'name':str(index)+"_m", 'object':cv2.merge([edge_mask, edge_mask, edge_mask])})
                self.record(data={'format':'.jpg', 'type':'defLn', 'name':str(index)+"_nm", 'object':cv2.merge([edge, edge, edge])})
                self.record(data={'format':'.txt', 'type':'defLn', 'name':str(index), 'object':line_def})
            
    def BLOB(self, img):
#        self.imgshow(img, "Scratch", obs=True)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        # I calculate the defects of area while removing the background indexe first, the and add it back to obtain tge correct array index
        area = np.sum(stats[np.where(stats[1:,cv2.CC_STAT_AREA] > self.def_area)[0]+1, cv2.CC_STAT_AREA])
        return area/img.size
    
    def detNoise(self, record=True):
        self.noise_metric = []
        noise_metric_append = self.noise_metric.append
        overall = []
        overall_append = overall.append
        for img in self.image:
            img_N = self.noiseFunc(img)
            self.imgshow(img_N, "Noise")
            overall_append(img_N)
        overall_append(np.uint8(np.around((overall[0] + overall[1] + overall[2])/3)))
        for index, noise in enumerate(overall):
            Noise_mean = np.sum(noise) / (np.sum(noise>0)+self.epison)
            Noise_ratio = np.sum(noise>0) / (noise.size+self.epison)
            metric = self.w0*Noise_mean+self.w1*Noise_ratio
            noise_metric_append(metric)
            if record and index < 3:
                self.record(data={'format':'.jpg', 'type':'Noise', 'name':str(index), 'object':cv2.merge([noise, noise, noise])})
                self.record(data={'format':'.txt', 'type':'Noise', 'name':str(index), 'object':metric})
        
    def noiseFunc(self, img):
        avg_img = cv2.blur(img,(3,3))
        kernel_h = np.array([[-1, 0, 1]], dtype=np.float32)
        kernel_v = np.array([[-1], [0], [1]], dtype=np.float32)
        D_h = cv2.filter2D(avg_img, -1, kernel_h)
        # add the mask 
        D_h[np.where(self.imgc.img_mask==255)] = 0
#        self.imgshow(D_h, name="noise")
        D_v = cv2.filter2D(avg_img, -1, kernel_v)
        # add the mask
        D_v[np.where(self.imgc.img_mask==255)] = 0
        
        N_cand = np.zeros_like(D_h, dtype=np.float32)
        # find the D_h<=D_hmean & D_v<=D_vmean position and then follow the paper to compare the D_h[position] and D_v[position]
        # put the value to the N_cand[position]
        # Do not use the for loop to complete bcoz of worse efficiency
        conditionPos = np.where((D_h>0) & (D_v>0))
        conditionVal = np.maximum(D_h[conditionPos], D_v[conditionPos])
        N_cand[conditionPos] = conditionVal
        N_candmean = np.mean(N_cand)
        N = np.zeros_like(N_cand, dtype=np.float32)
        N[np.where(N_cand>N_candmean)] = N_cand[np.where(N_cand>N_candmean)]
        return N

    def detDef(self):
        self.defect_ratio = []
        defect_ratio_append = self.defect_ratio.append
        overall = []
        overall_append = overall.append
        for index, img in enumerate(self.image):
            blur_img = cv2.GaussianBlur(img,(3,3),0)
            # if index==2 -> coaxial light source:do the sobel fisrt 
            if index == 2:
                thresh = int(np.mean(blur_img) - 1*np.std(blur_img))
                binary_img = cv2.threshold(blur_img, thresh, 255, cv2.THRESH_BINARY_INV)[1]
                binary_img = cv2.medianBlur(binary_img, 3)
                overall.append(binary_img)
            else:
                thresh = int(np.mean(blur_img) + 3*np.std(blur_img))
                binary_img = cv2.threshold(blur_img, thresh, 255, cv2.THRESH_BINARY)[1]        
                overall_append(binary_img)
            result = self.imgc.img_mask & binary_img
#            self.imgshow(result, "Defects")
#            print("result_white:{} mask_white:{}".format(np.sum(result==255), np.sum(self.imgc.img_mask==255)))
            ratio = np.sum(result==255) / np.sum(self.imgc.img_mask==255)
#            print(ratio)
            defect_ratio_append(ratio)
        OVERALL = overall[0] | overall[1] | overall[2]
        result = self.imgc.img_mask & OVERALL
        ratio = np.sum(result==255) / np.sum(self.imgc.img_mask==255)
#        print("result_white:{} mask_white:{}".format(np.sum(result==255), np.sum(self.imgc.img_mask==255)))
#        print(ratio)
        defect_ratio_append(ratio)
    
    def rewFunc(self):
        self.done[0] = 0
        for index in range(self.state.shape[0]):
            # the last two states are the overall results
            if index%2:
                r  = np.exp(self.state[index-1]*100)-1
                r -= np.exp(self.state[index]*0)-1
                self.reward[index-1] = r
                self.reward[index] = r
                
        # re-allocate the total reward
        indices = [i for i in range(self.reward.shape[0]-2) if i%2 ]
        alloc_ratio = [self.reward[ind]/self.reward[indices].sum() for ind in indices]
        for ind, ratio in zip(indices, alloc_ratio):
            self.reward[ind] += self.reward[-1]*ratio
            self.reward[ind-1] += self.reward[-1]*ratio
        

        self.total_reward = np.sum(self.reward[:-2])/2
        if self.total_reward > self.maxReward:
            self.goalCounter += 1
            self.maxReward = self.total_reward
            self.maxpoint = self.step
            self.reward *= 1.1
            if self.goalCounter > 4:
                self.done[0] = 2
        elif self.total_reward <= 0:
            self.reward[np.where(self.reward[:-2]<0)] *= 1.1
            self.minCounter += 1
            if self.minCounter > 4:
                self.done[0] = 3

        print("maxReward:{:+.4e}\tcurrReward:{:+.4e}".format(self.maxReward, self.total_reward))
        print("maxCounter:{}".format(self.goalCounter))
        print("minCounter:{}".format(self.minCounter))
                    
    def record(self, data=None):
        d = [data['format'], data['type'], data['name']]
        if not d in self.recData:
            self.recData.append(d)
            self.recCounter[d[1]+d[2]] = 0
        else:
            self.recCounter[d[1]+d[2]] += 1
        if d[0] == '.jpg':
            father_dir = os.path.join(self.savedir,d[1])
            if not os.path.isdir(father_dir):
                os.mkdir(father_dir) 
            child_dir = os.path.join(father_dir, d[2])
            if not os.path.isdir(child_dir):
                os.mkdir(child_dir)
            cv2.imwrite(os.path.join(child_dir, str(self.recCounter[d[1]+d[2]])) + d[0], data['object'])
            
        else:
            with open(self.savedir+d[1]+d[2]+d[0], "a+") as f:
                f.write("%s\n" % str(data['object']))
        
        
    def imgshow(self, img, name="show", obs=False, pos=None):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if not pos is None:
            cv2.moveWindow(name, pos[0], pos[1])
        cv2.imshow(name, img)
        if not obs:
            cv2.waitKey(1)
        else:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

 

            
"""""""""""""""
Test code (metric):
imgp = imageprocess(s_dim=8, a_dim=6, crop_ratio=0.7, noise_param=[0.2, 0.8])
inputFile = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/Image/"
allImgName = glob.glob(inputFile + "*.jpg")[:3]
IMG = []
IMG_append = IMG.append
for name in allImgName:
    img = cv2.imread(name, 0)
    IMG_append(img)
IMG = IMG[:3]
state, reward, done = imgp.parse(IMG)
print(state)
print(reward)
print(done)

"""""""""""""""

"""""""""""""""
Test code 2:

if __name__ == '__main__':
    t1 = time.time()
    imgr = imgReward(s_dim=8, a_dim=6, crop_ratio=0.05, ang_tol=3, maskSize=15, noise_ratio=0.5, def_area = 20, lines = 50)
    
    print("duration:{:.3f} s".format(time.time()-t1))

"""""""""""""""
"""""""""""""""
Test code 3:

if __name__ == '__main__':
    func.state(mode=1)
    imgr = imgReward(s_dim=8, a_dim=6, crop_ratio=0.05, ang_tol=3, maskSize=15, noise_ratio=0.5, def_area = 20, lines = 50)
    light = [0, 0, 0, 0]
    img_pool = []
    D = []
    l = np.array([30, 150, 255])
    e = np.array([3, 11, 19])
    le = np.array(np.meshgrid(l, e, l, e, l, e), dtype=np.float32).T.reshape(-1, 6)
    for command in le:
        for channel in range(3):
            light[channel] = int(command[channel*2])
            func.action(expTime = command[channel*2+1], light=light, channel=channel)    
            img_pool.append(func.feedback())
            light[channel] = 0
        A = imgr.parse(img_pool, times=0, rec=False)
        D.append(imgr.D)
        img_pool = []
    
    d = np.asarray(D)
"""""""""""""""