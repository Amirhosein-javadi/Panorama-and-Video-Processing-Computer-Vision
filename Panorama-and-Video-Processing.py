import  numpy as np
import  cv2
import os
import scipy
from scipy import signal
import sys
import operator

def Find_Homography(Img1,Img2):
    sift = cv2.SIFT_create()
    KeyPoint1,Desc1 = sift.detectAndCompute(Img1,None)
    KeyPoint2,Desc2 = sift.detectAndCompute(Img2,None)
    FLANN_INDEX_KDTREE = 1
    param1 = dict(algorithm=FLANN_INDEX_KDTREE, trees=20)
    param2 = dict(checks=100)
    flann = cv2.FlannBasedMatcher(param1,param2)
    matches = flann.knnMatch(Desc1,Desc2,k=2)
    Goodpoints = []
    for i,j in matches:
        if i.distance < 0.75*j.distance:
            Goodpoints.append(i)
    Pts1 = np.int32([ KeyPoint1[i.queryIdx].pt for i in Goodpoints ]).reshape(-1,2)
    Pts2 = np.int32([ KeyPoint2[i.trainIdx].pt for i in Goodpoints ]).reshape(-1,2)
    M, mask = cv2.findHomography(Pts1,Pts2,cv2.RANSAC,10,100)
    return M

def Find_Boundary(im1,im2,mask1,mask2):
    mask = mask1 * mask2
    index = np.where(mask==1)
    minrow = np.min(index[0])
    maxrow = np.max(index[0])
    mincol = np.min(index[1])
    maxcol = np.max(index[1])  
    diff   = np.sum(np.abs(im1[minrow:maxrow+1,mincol:maxcol+1]-im2[minrow:maxrow+1,mincol:maxcol+1]),axis=2)
    diff = ((mask[minrow:maxrow+1,mincol:maxcol+1]*diff)+(1-mask[minrow:maxrow+1,mincol:maxcol+1])*500).T
    bestcut = Find_Best_Cut(diff)
    bestcut = np.array(list(bestcut.T)*(maxcol-mincol+1)).T
    newim = np.zeros([maxrow-minrow+1,maxcol-mincol+1,3])
    horizontal_idx = np.indices(newim[:,:,0].shape)[1]
    best_cut = bestcut <= horizontal_idx
    return best_cut,minrow,maxrow,mincol,maxcol

def Find_Best_Cut(diffrences):
    row,col = diffrences.shape
    ways    = np.zeros([row,col]).astype(np.uint16)
    E_first = np.zeros(row)
    E_last  = np.zeros(row)
    for j in range(col-1):
        for i in range(row):
            if i==0:
                ways[i,j]  = np.where(diffrences[i:i+2,j]==np.min(diffrences[i:i+2,j]))[0][0] 
                E_first[i]   = np.min(diffrences[i:i+2,j]) + E_last[ways[i,j]] 
            elif i==col-1 :
                ways[i,j] = np.where(diffrences[i-1:i+1,j]==np.min(diffrences[i-1:i+1,j]))[0][0] + i - 1
                E_first[i]  = np.min(diffrences[i-1:i+1,j]) + E_last[ways[i,j]]
            else:
                ways[i,j] = np.where(diffrences[i-1:i+2,j]==np.min(diffrences[i-1:i+2,j]))[0][0] + i - 1
                E_first[i]  = np.min(diffrences[i-1:i+2,j]) + E_last[ways[i,j]]
        E_last = np.copy(E_first)
    j = col-1    
    for i in range(row):
        ways[i,j] = i
        E_first[i] = diffrences[i,j] + E_last[ways[i,j]]
    minimum_way = np.where(E_first==np.min(E_first))[0][0]
    Best_Cut = np.zeros([col,1]).astype(np.int16)
    Best_Cut[-1] = ways[minimum_way,col-1]
    for i in range(2,col+1):
        Best_Cut[col-i] = ways[Best_Cut[-i+1],-i] 
    return Best_Cut

def Create_Background(rownum,colnum,flag,deltax,k0,Background):
    flagcopy = flag
    newk0 = k0
    innerflag = False
    for i in range(rownum):
       for j in range(colnum):
           globals()["pixel" + str(i) + "and" + str(j)] = {} 
    if deltax<5500:        
        jump = 5 
    else:
        jump = 1        
    for k in range(k0,901,jump):
        filename = "warped/frame-%d.jpg"%(k)  
        Img = cv2.imread(filename)
        state = ~((Img[0:rownum,deltax:deltax+colnum,0]<10)*(Img[0:rownum,deltax:deltax+colnum,1]<10)*(Img[0:rownum,deltax:deltax+colnum,2]<10))
        state = np.sum(state)
        if state == 0:
            if flagcopy == True:
                continue
            else:
                break
        else:
            flagcopy = False;
            if flag == False:
                flag = True
            if innerflag == False:
                newk0 = k
                innerflag = True
            for i in range(rownum):
                for j in range(colnum):
                    state = ~((Img[i,j+deltax,0]<10)*(Img[i,j+deltax,1]<10)*(Img[i,j+deltax,2]<10))
                    if state == False:
                        continue
                    else:
                        key = str(Img[i,j+deltax,0])+' '+str(Img[i,j+deltax,1])+' '+str(Img[i,j+deltax,2])
                        if key not in globals()["pixel"+str(i)+"and"+str(j)]:
                            globals()["pixel"+str(i)+"and"+str(j)][key] = 1;
                        else:
                            globals()["pixel"+str(i)+"and"+str(j)][key] += 1;   
    for i in range(rownum):
        for j in range(colnum):
            if (bool(globals()["pixel"+str(i)+"and"+str(j)])==False):
                Background[i,j+deltax,:] = [0,0,0]
            else:
                maxval = max(globals()["pixel"+str(i)+"and"+str(j)].items(), key=operator.itemgetter(1))[0]
                maxval = np.array(maxval.split())
                Background[i,j+deltax,0] = maxval[0]
                Background[i,j+deltax,1] = maxval[1]
                Background[i,j+deltax,2] = maxval[2] 
    for i in range(rownum):
        for j in range(colnum):
            globals()["pixel"+str(i)+"and"+str(j)].clear()                   
    return Background,newk0,flag

# os.system('ffmpeg -i video.mp4 -r 30 -t 00:00:30 frames/frame-%d.jpg') #$filename%03d.jpg

# Img270 = cv2.imread("frames/frame-270.jpg")         
# Img450 = cv2.imread('frames/frame-450.jpg') 
# row,col,dim = Img270.shape
# M = Find_Homography(Img270,Img450)
# M_inverse = np.linalg.inv(M)
# Red_rectangle = cv2.rectangle(np.zeros_like(Img450), (500,500), (500+300,500+300), (0, 0, 255) , 2) 
# Img450Rect = np.zeros_like(Img450)
# Img450Rect[:,:,0] = (Red_rectangle[:,:,2]>0)*Red_rectangle[:,:,0] + ~(Red_rectangle[:,:,2]>0) * Img450[:,:,0];
# Img450Rect[:,:,1] = (Red_rectangle[:,:,2]>0)*Red_rectangle[:,:,1] + ~(Red_rectangle[:,:,2]>0) * Img450[:,:,1];
# Img450Rect[:,:,2] = (Red_rectangle[:,:,2]>0)*Red_rectangle[:,:,2] + ~(Red_rectangle[:,:,2]>0) * Img450[:,:,2];
# cv2.imwrite ('res01-450-rect.jpg',Img450Rect)
# Warped_Red_rectangle =cv2.warpPerspective(Red_rectangle,M_inverse,(1920,1080))
# Img270Rect = np.zeros_like(Img270)
# Img270Rect[:,:,0] = (Warped_Red_rectangle[:,:,2]>0)*Warped_Red_rectangle[:,:,0] + ~(Warped_Red_rectangle[:,:,2]>0) * Img270[:,:,0];
# Img270Rect[:,:,1] = (Warped_Red_rectangle[:,:,2]>0)*Warped_Red_rectangle[:,:,1] + ~(Warped_Red_rectangle[:,:,2]>0) * Img270[:,:,1];
# Img270Rect[:,:,2] = (Warped_Red_rectangle[:,:,2]>0)*Warped_Red_rectangle[:,:,2] + ~(Warped_Red_rectangle[:,:,2]>0) * Img270[:,:,2];
# cv2.imwrite ('res02-270-rect.jpg',Img270Rect)
############################################  Section 1
# xscale      = 4
# yscale      = 3
# newrow = np.uint16(yscale*row)
# newcol = np.uint16(xscale*col)
# Big270      = np.zeros([newrow,newcol,3]).astype(np.uint8)
# Mask270     = np.zeros([newrow,newcol]).astype(np.uint8)
# Big270 [newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img270
# Mask270[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2] = 1;
# Big450      =  np.zeros([newrow,newcol,3]).astype(np.uint8)
# Mask450     = np.zeros([newrow,newcol]).astype(np.uint8)
# Big450 [newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img450
# filename = "warped/frame-450.jpg"
# cv2.imwrite(filename, Big450)
# Final_im    = Big450.copy()
# Mask450[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2]  = 1;
# M270        = Find_Homography(Big270,Big450)
# Warped270   = cv2.warpPerspective(Big270,M270,(newcol,newrow))
# filename = "warped/frame-270.jpg"
# cv2.imwrite(filename, Warped270)
# Mask270     = cv2.warpPerspective(Mask270,M270,(newcol,newrow))
# state = (Final_im[:,:,0]==0)*(Final_im[:,:,1]==0)*(Final_im[:,:,2]==0)
# Final_im[:,:,0]    = Final_im[:,:,0] + Warped270[:,:,0]  * state
# Final_im[:,:,1]    = Final_im[:,:,1] + Warped270[:,:,1]  * state
# Final_im[:,:,2]    = Final_im[:,:,2] + Warped270[:,:,2]  * state
# state = (Final_im[:,:,0]>0)*(Final_im[:,:,1]>0)*(Final_im[:,:,2]>0)
# minrow = np.min(np.where(state)[0])
# maxrow = np.max(np.where(state)[0])
# mincol = np.min(np.where(state)[1])
# maxcol = np.max(np.where(state)[1]) 
# cv2.imwrite('res03-270-450-panorama.jpg',Final_im[minrow:maxrow+1,mincol:maxcol])
##############################################  Section 2
# Bestcut,minrow,maxrow,mincol,maxcol  = Find_Boundary(Warped270,Big450,Mask270,Mask450)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,0] = Warped270[minrow:maxrow+1,mincol:maxcol+1,0]*(~Bestcut) + Final_im[minrow:maxrow+1,mincol:maxcol+1,0]*(Bestcut)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,1] = Warped270[minrow:maxrow+1,mincol:maxcol+1,1]*(~Bestcut) + Final_im[minrow:maxrow+1,mincol:maxcol+1,1]*(Bestcut)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,2] = Warped270[minrow:maxrow+1,mincol:maxcol+1,2]*(~Bestcut) + Final_im[minrow:maxrow+1,mincol:maxcol+1,2]*(Bestcut)
##############################################  Section 2
# Img90 = cv2.imread('frames/frame-90.jpg') 
# Big90 = np.zeros([newrow,newcol,3]).astype(np.uint8)
# Big90[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img90
# Mask90  = np.zeros([newrow,newcol]).astype(np.uint8)
# Mask90[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2] = 1;
# M90 = Find_Homography(Big90,Warped270)
# Warped90 = cv2.warpPerspective(Big90,M90,(newcol,newrow))
# filename = "warped/frame-90.jpg"
# cv2.imwrite(filename, Warped90)
# Mask90 = cv2.warpPerspective(Mask90,M90,(newcol,newrow))
# state = (Final_im[:,:,0]==0)*(Final_im[:,:,1]==0)*(Final_im[:,:,2]==0)
# Final_im[:,:,0]    = Final_im[:,:,0] + Warped90[:,:,0]  * state
# Final_im[:,:,1]    = Final_im[:,:,1] + Warped90[:,:,1]  * state
# Final_im[:,:,2]    = Final_im[:,:,2] + Warped90[:,:,2]  * state
# Bestcut,minrow,maxrow,mincol,maxcol  = Find_Boundary(Warped90,Warped270,Mask90,Mask270)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,0] = Warped90[minrow:maxrow+1,mincol:maxcol+1,0]*(~Bestcut) + Final_im[minrow:maxrow+1,mincol:maxcol+1,0]*(Bestcut)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,1] = Warped90[minrow:maxrow+1,mincol:maxcol+1,1]*(~Bestcut) + Final_im[minrow:maxrow+1,mincol:maxcol+1,1]*(Bestcut)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,2] = Warped90[minrow:maxrow+1,mincol:maxcol+1,2]*(~Bestcut) + Final_im[minrow:maxrow+1,mincol:maxcol+1,2]*(Bestcut)
##############################################  Section 2
# Img630 = cv2.imread('frames/frame-630.jpg') 
# Big630 = np.zeros([newrow,newcol,3]).astype(np.uint8)
# Big630[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img630
# Mask630  = np.zeros([newrow,newcol]).astype(np.uint8)
# Mask630[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2] = 1;
# M630 = Find_Homography(Big630,Big450)
# Warped630 = cv2.warpPerspective(Big630,M630,(newcol,newrow))
# filename = "warped/frame-630.jpg"
# cv2.imwrite(filename, Warped630)
# Mask630 = cv2.warpPerspective(Mask630,M630,(newcol,newrow))
# state = (Final_im[:,:,0]==0)*(Final_im[:,:,1]==0)*(Final_im[:,:,2]==0)
# Final_im[:,:,0]    = Final_im[:,:,0] + Warped630[:,:,0]  * state
# Final_im[:,:,1]    = Final_im[:,:,1] + Warped630[:,:,1]  * state
# Final_im[:,:,2]    = Final_im[:,:,2] + Warped630[:,:,2]  * state
# Bestcut,minrow,maxrow,mincol,maxcol = Find_Boundary(Big450,Warped630,Mask450,Mask630)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,0] = Final_im[minrow:maxrow+1,mincol:maxcol+1,0]*(~Bestcut) + Warped630[minrow:maxrow+1,mincol:maxcol+1,0]*(Bestcut)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,1] = Final_im[minrow:maxrow+1,mincol:maxcol+1,1]*(~Bestcut) + Warped630[minrow:maxrow+1,mincol:maxcol+1,1]*(Bestcut)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,2] = Final_im[minrow:maxrow+1,mincol:maxcol+1,2]*(~Bestcut) + Warped630[minrow:maxrow+1,mincol:maxcol+1,2]*(Bestcut)
##############################################  Section 2
# Img810 = cv2.imread('frames/frame-810.jpg') 
# Big810 = np.zeros([newrow,newcol,3]).astype(np.uint8)
# Big810[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img810
# Mask810  = np.zeros([newrow,newcol]).astype(np.uint8)
# Mask810[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2] = 1;
# M810 = Find_Homography(Big810,Warped630)
# Warped810 = cv2.warpPerspective(Big810,M810,(newcol,newrow))
# filename = "warped/frame-810.jpg"
# cv2.imwrite(filename, Warped810)
# Mask810 = cv2.warpPerspective(Mask810,M810,(newcol,newrow))
# Final_im[:,:,0]    = Final_im[:,:,0] + Warped810[:,:,0]  * state
# Final_im[:,:,1]    = Final_im[:,:,1] + Warped810[:,:,1]  * state
# Final_im[:,:,2]    = Final_im[:,:,2] + Warped810[:,:,2]  * state
# Final_im = Final_im + Warped810 * (Final_im==0)
# Bestcut,minrow,maxrow,mincol,maxcol = Find_Boundary(Warped630,Warped810,Mask630,Mask810)
# Final_im[minrow:maxrow+1,mincol:maxcol+1,0] = Final_im[minrow:maxrow+1,mincol:maxcol+1,0]*(~Bestcut) + Warped810[minrow:maxrow+1,mincol:maxcol+1,0]*(Bestcut) + (Warped810[minrow:maxrow+1,mincol:maxcol+1,0]==0)*(Bestcut)*Final_im[minrow:maxrow+1,mincol:maxcol+1,0]
# Final_im[minrow:maxrow+1,mincol:maxcol+1,1] = Final_im[minrow:maxrow+1,mincol:maxcol+1,1]*(~Bestcut) + Warped810[minrow:maxrow+1,mincol:maxcol+1,1]*(Bestcut) + (Warped810[minrow:maxrow+1,mincol:maxcol+1,1]==0)*(Bestcut)*Final_im[minrow:maxrow+1,mincol:maxcol+1,1]
# Final_im[minrow:maxrow+1,mincol:maxcol+1,2] = Final_im[minrow:maxrow+1,mincol:maxcol+1,2]*(~Bestcut) + Warped810[minrow:maxrow+1,mincol:maxcol+1,2]*(Bestcut) + (Warped810[minrow:maxrow+1,mincol:maxcol+1,2]==0)*(Bestcut)*Final_im[minrow:maxrow+1,mincol:maxcol+1,2]
# state = (Final_im[:,:,0]>0)*(Final_im[:,:,1]>0)*(Final_im[:,:,2]>0)
# minrow = np.min(np.where(state)[0])
# maxrow = np.max(np.where(state)[0])
# mincol = np.min(np.where(state)[1])
# maxcol = np.max(np.where(state)[1]) 
# cv2.imwrite('res04-key-frames-panorama.jpg',Final_im[minrow:maxrow+1,mincol:maxcol])
###########################################  Section 3 
# Img90 = cv2.imread("warped/frame-90.jpg") 
# Img270 = cv2.imread("frames/frame-270.jpg") 
# row,col,dim = Img270.shape
# xscale      = 4
# yscale      = 3
# newrow = np.uint16(yscale*row)
# newcol = np.uint16(xscale*col)
# for i in range(1,90):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img90)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i)
#     cv2.imwrite(filename,Warped)
# for i in range(91,181):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img90)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i) 
#     cv2.imwrite(filename,Warped)
############################################ Section 3 
# Img270 = cv2.imread("warped/frame-270.jpg") 
# for i in range(181,270):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img270)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i)
#     cv2.imwrite(filename,Warped)
# for i in range(271,361):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img270)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i) 
#     cv2.imwrite(filename,Warped)
#############################################  Section 3  
# Img450 = cv2.imread("warped/frame-450.jpg") 
# for i in range(361,450):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img450)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i)
#     cv2.imwrite(filename,Warped)
# for i in range(450,541):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img450)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i)     
#     cv2.imwrite(filename,Warped)
##############################################  Section 3        
# Img630 = cv2.imread("warped/frame-630.jpg") 
# for i in range(600,630):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img630)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i)
#     cv2.imwrite(filename,Warped)
# for i in range(631,721):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img630)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i) 
#     cv2.imwrite(filename,Warped)
# ################################################  Section 3      
# Img810 = cv2.imread("warped/frame-810.jpg") 
# for i in range(721,810):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img810)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i)
#     cv2.imwrite(filename,Warped)
# for i in range(811,901):
#     filename = "frames/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     BigImg = np.zeros([newrow,newcol,3]).astype(np.uint8)
#     BigImg[newrow//2-row//2:newrow//2+row//2,newcol//2-col//2:newcol//2+col//2,:] = Img
#     M = Find_Homography(BigImg,Img810)
#     Warped = cv2.warpPerspective(BigImg,M,(newcol,newrow))
#     filename = "warped/frame-%d.jpg"%(i)  
#     cv2.imwrite(filename,Warped)
# os.system('ffmpeg -i warped/frame-%d.jpg -r 15 res05-reference-plane.mp4')

# ################################################  Section 4
# Img270 = cv2.imread("frames/frame-270.jpg") 
# row,col,dim = Img270.shape
# xscale      = 4
# yscale      = 3
# newrow = np.uint16(yscale*row)
# newcol = np.uint16(xscale*col)
# Background = np.zeros([newrow,newcol,3])
# flag = False
# k0 = 1
# for i in range(1,newcol-160,160):
#     Background,k0,flag = Create_Background(newrow,160,flag,i,k0,Background)
#     cv2.imwrite('res06-background-panorama.jpg',Background) 
# cv2.imwrite('res06-background-panorama.jpg',Background)  
# # ###############################################     Section 5
# Background = cv2.imread('res06-background-panorama.jpg')
# for i in range(1,901):
#     filename = "warped/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     state = ~((Img[:,:,0]==0)*(Img[:,:,1]==0)*(Img[:,:,2]==0))
#     frame = np.zeros_like(Img)
#     frame[:,:,0] = state[:,:]*Background[:,:,0]
#     frame[:,:,1] = state[:,:]*Background[:,:,1]
#     frame[:,:,2] = state[:,:]*Background[:,:,2]
#     filename = "background/frame-%d.jpg"%(i)
#     cv2.imwrite(filename,frame)
# os.system('ffmpeg -i background/frame-%d.jpg -r 15 res07-background-video.mp4')
# # # ###############################################   Section 6
# threshold = 65;
# kernel =np.ones([50,50]).astype(np.uint8)
# for i in range(1,901):
#     filename = "warped/frame-%d.jpg"%(i)
#     Img1 = cv2.imread(filename)
#     hsv1 = cv2.cvtColor(Img1,cv2.COLOR_BGR2HSV).astype(np.int16)
#     filename = "background/frame-%d.jpg"%(i)
#     Img2 = cv2.imread(filename) 
#     hsv2 = cv2.cvtColor(Img2,cv2.COLOR_BGR2HSV).astype(np.int16)
#     state = (~((Img1[:,:,0]==0)*(Img1[:,:,1]==0)*(Img1[:,:,2]==0))*(~((Img2[:,:,0]==0)*(Img2[:,:,1]==0)*(Img2[:,:,0]==2)))).astype(np.uint8)
#     state = cv2.erode(state,kernel,iterations = 1)
#     diff = np.abs(hsv2[:,:,2]-hsv1[:,:,2])*state
#     diff = cv2.medianBlur(diff,5)
#     state = diff>threshold
#     Img1[:,:,2] = (state*255) + (~state)*Img1[:,:,2]
#     filename = "foreground/frame-%d.jpg"%(i)   
#     cv2.imwrite(filename,Img1)
# os.system('ffmpeg -i foreground/frame-%d.jpg -r 15 res08-foreground-video.mp4')  

# ###############################################   Section 7
# Img270 = cv2.imread("frames/frame-270.jpg") 
# row,col,dim = Img270.shape
# xscale      = 4
# yscale      = 3
# newrow = np.uint16(yscale*row)
# newcol = np.uint16(xscale*col)
# bigerxscale      = 4*1.5
# bigernewcol = np.uint16(bigerxscale*col)
# for i in range(1,901):
#     deltax = 1910/450 * i - (1910)
#     M = np.array([[1,0,deltax],
#                   [0,1,0    ],
#                   [0,0,1    ]]).astype(np.float64)
#     filename = "background/frame-%d.jpg"%(i)
#     Img = cv2.imread(filename) 
#     Bigim = np.zeros([newrow,bigernewcol,3]).astype(np.uint8)
#     Bigim[:,bigernewcol//2-newcol//2:bigernewcol//2+newcol//2,:]=Img
#     Shiftim = cv2.warpPerspective(Bigim,M,(bigernewcol,newrow))
#     filename = "wide/frame-%d.jpg"%(i)
#     cv2.imwrite(filename,Shiftim)
    
# os.system('ffmpeg -i wide/frame-%d.jpg -r 30 -vf scale=4096:-2,format=yuv420p res09-background-video-wider.MP4') 

os.system('ffmpeg -i wide/frame-%d.jpg -r 15 res09-background-video-wider.mp4')  