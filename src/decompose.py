'''
Created on 2010-5-31

@author: JuanLu
Decompose a single skin image into melanin component image and haemoglobin component image by using independent component analysis
'''
from PIL import Image
from numpy import *

C_e=array([[0.0246, 0.0316, 0.0394], [0.0193,0.0755,0.0666]]) # the two vectores correspond to melanin component and haemoglobin components respectively
C_b=array([[1.1601, 2.8347]]) # C_b is a stationary vector caused by other pigments and skin structure

C_e=C_e.transpose()
C_b=C_b.transpose()

im=Image.open("./images/sample.bmp")
im_array=asarray(im)
print(im_array.shape)
im_arrayF=im_array.astype('float')
im_log=log(im_arrayF/255)
size=im.size
im_r=size[0]
im_c=size[1]

#convert the skin image into a log colour space
log_r=im_log[:,:,0]
log_g=im_log[:,:,1]
log_b=im_log[:,:,2]

col_r=log_r.flatten()
col_g=log_g.flatten()
col_b=log_b.flatten()

N=col_r.shape[0]
C=array([-col_r,-col_g,-col_b])

#concver the skin image from the log colour space into a space composed of melain and haemoglobin components
C_add=linalg.pinv(C_e)
cc=dot(C_add,C)
ima_b=tile(C_b, (1,N))
q=cc-ima_b

#extract the melanin component image
Pig1=dot(C_e,diag([1,0])) 
Pig1=dot(Pig1,q)
I1=exp(-Pig1.transpose())
I1_im=I1.reshape(im_c,im_r,3).copy()
I1_im=I1_im * 255
I1_im=I1_im.clip(0,255)
print(I1_im.shape)
I1_im_result=Image.fromarray(uint8(I1_im))
I1_im_result.save('./images/melanin.bmp')

#extract the haemoglobin component image
Pig2=dot(C_e,diag([0,1]))
Pig2=dot(Pig2,q)
I2=exp(-Pig2.transpose())
I2_im=I2.reshape(im_c,im_r,3).copy()
I2_im=I2_im * 255
I2_im=I2_im.clip(0,255)
print(I2_im.shape)
I2_im_result=Image.fromarray(uint8(I2_im))
#I1_im_result=Image.fromarray(I1_im)
I2_im_result.save('./images/haemoglobin.bmp')

#show the result
im.show()
I1_im_result.show() #show the melanin component image
I2_im_result.show() #show the haemoglobin component image





