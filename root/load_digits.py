import os
import cv2
import numpy as np
import pdb #mz

img0 = cv2.imread('data/digits.png',0) #0: gray
img = cv2.bilateralFilter(img0,9,75,75)
ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY); 
#cv2.imwrite('data/digits_thresh.png',img);

delta=img.shape[1]//10
height=np.arange(0,img.shape[0],delta)
width=np.arange(0,img.shape[1],delta)
#pdb.set_trace() #mz

for h in height:
    ran=np.arange(h-5,min(h+5,img.shape[0]))
    img[ran,:]=0
    
for w in width:
    ran=np.arange(w-5,min(w+5,img.shape[1]))
    img[:,ran]=0

#cv2.imwrite('data/digits_partioned.png',img)

data_sc=[[],[],[],[],[],[],[],[],[],[],[]]
data_0, data_T, data_c, data_cSq, data_150, labels = [], [], [], [], [], []

for h_ind, h in enumerate(height[0:15]):
    for w_ind, w in enumerate(width[0:10]):
        labels.append(w_ind)
        image0=img0[h+5:height[h_ind+1]-5,w+5:width[w_ind+1]-5]
        image=img[h+5:height[h_ind+1]-5,w+5:width[w_ind+1]-5]
        img_p=np.copy(image)
        indh=np.argwhere(np.amin(image,axis=0)==0) #shape[1] ver.
        indv=np.argwhere(np.amin(image,axis=1)==0) #shape[0] hor.
        #pdb.set_trace() #mz
        image[:,0:indh[0,0]]=0
        image[:,indh[-1,0]+1:]=0

        image[0:indv[0,0],:]=0
        image[indv[-1,0]+1:,:]=0
        img[h+5:height[h_ind+1]-5,w+5:width[w_ind+1]-5]=image

        #pdb.set_trace() #mz
        img_c0=img_p[indv[0,0]:indv[-1,0],indh[0,0]:indh[-1,0]]

        ch=img_c0.shape[0]; cw=img_c0.shape[1]; d=max(ch,cw);
        img_cSq=255*np.ones([d,d])
        img_cSq[(d-ch)//2:(d-ch)//2+ch,(d-cw)//2:(d-cw)//2+cw]=img_c0

        d1=int(1.5*d); 
        img_150=255*np.ones([d1,d1])
        img_150[(d1-ch)//2:(d1-ch)//2+ch,(d1-cw)//2:(d1-cw)//2+cw]=img_c0

        for ks, scale in enumerate(np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2])):

            ds=int(scale*d)
            img_sc=255*np.ones([ds,ds])
            img_sc[(ds-ch)//2:(ds-ch)//2+ch,(ds-cw)//2:(ds-cw)//2+cw]=img_c0
            data_sc[ks].append(255-cv2.resize(img_sc, (28,28), interpolation= cv2.INTER_AREA))

        data_0.append(255-cv2.resize(image0, (28,28), interpolation= cv2.INTER_AREA)) 

        data_T.append(255-cv2.resize(img_p, (28,28), interpolation= cv2.INTER_AREA))

        data_c.append(255-cv2.resize(img_c0, (28,28), interpolation= cv2.INTER_AREA)) 

        data_cSq.append(255-cv2.resize(img_cSq, (28,28), interpolation= cv2.INTER_AREA))

        data_150.append(255-cv2.resize(img_150, (28,28), interpolation= cv2.INTER_AREA))


labels=np.array(labels)
np.save('data/labels.npy',labels)

data=np.array([data_0, data_T, data_c, data_cSq, data_150])
np.save('data/data.npy',np.array(data))

np.save('data/data_sc.npy',np.array(data_sc))


#pdb.set_trace() #mz
