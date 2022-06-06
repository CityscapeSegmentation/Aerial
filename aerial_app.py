import streamlit as st
from datetime import time, datetime
import torch
import matplotlib
import numpy as np
from  PIL import Image
import cv2

from model import UNet

from utils import  givin_colors
from utils import  target_names
from utils import  AddTextToMask
from sklearn.metrics import classification_report
import pandas as pd
import os
from  utils import aerial_transforms

path='data/val/rgb'

if 'imgList' not in st.session_state:	
	st.session_state.imgList = os.listdir(path)


	


if 'count' not in st.session_state:
	st.session_state.count = 1

if 'model' not in st.session_state:
	#try:
		#st.session_state.model = UNet(15)	
		deep_model= UNet(n_channels=3, n_classes=6, bilinear=True)
		deep_model.cpu()
		deep_model.load_state_dict(torch.load('weights/unet_cpu.pt',map_location ='cpu'))
		st.session_state.flag=True
		st.session_state.model=deep_model
	#except Exception as e:
	#	print(e)
else:
	deep_model=st.session_state.model

	

#uploaded_file = st.file_uploader("Choose a file")

col1, col2 = st.columns(2)


# increment = st.button('Next')
# if increment:
#     st.session_state.count += 1

# # A button to decrement the counter
# decrement = st.button('Decrement')
# if decrement:
#     st.session_state.count -= 1


with col1:
   decrement = st.button('Prev')
   if decrement:
       st.session_state.count -= 1
   if st.session_state.count<1:
       st.session_state.count=1

with col2:
   increment = st.button('Next')
   if increment:
       st.session_state.count += 1
   if st.session_state.count>72:
      st.session_state.count=72

target=int(st.session_state.count)
file_name=st.session_state.imgList[ target-1 ]

rgb_path='data/val/rgb/'+file_name
mask_path='data/val/mask/'+file_name


st.write(rgb_path)
  
rgb=Image.open(rgb_path)

st.write(rgb.size)

#image=np.array(rgb)
image = aerial_transforms(rgb)


image=np.array(image)
image=cv2.resize(image,(512,512))
image=image/255.0

image=np.moveaxis(image,-1,0)

image=torch.tensor(image)
image=image.float()




st.write(image.shape)





image=torch.unsqueeze(image, 0).cpu()
#deep_model.cpu()

preds=deep_model(image)

values,indecies=torch.max(preds,dim=1)

indecies=indecies.squeeze().cpu().numpy()







mask=Image.open(mask_path)

mask=np.array(mask,dtype=np.uint8)

mask=cv2.resize(mask,(512,512))

shape=mask.shape
with col1:
   st.image(rgb, caption=str(target)+'.png')
with col2:
 
   colored_mask=AddTextToMask(mask,target_names)
   st.image(colored_mask, caption=' Mask_'+str(target)+'.png')

st.write(indecies.shape)

pred=np.array(indecies,dtype=np.uint8)


#colored_pred=givin_colors[colored_pred]
#colored_pred=colored_pred.reshape((shape[0],shape[1],3))
colored_pred=AddTextToMask(pred,target_names)
	
st.image(colored_pred, caption=' Preds'+str(target)+'.png')


print('mask shape',mask.shape)
print('pred shape',pred.shape)

for i in range(6):
   pred[0,i]=i



#st.write(classification_report(mask.reshape(-1), pred.reshape(-1), target_names=target_names))

#print(classification_report(mask.reshape(-1), pred.reshape(-1),target_names=target_names)        )

report_dict=classification_report(mask.reshape(-1), pred.reshape(-1),target_names=target_names, output_dict=True)


df=pd.DataFrame(report_dict)

df1=df.T

st.dataframe(df1)

 
 


