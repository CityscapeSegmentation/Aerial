


from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
import streamlit as st

import numpy as np
import cv2
from PIL import ImageDraw,ImageFont,Image







target_names = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']




# tmp=[
# [0,0,0],
# [0,0,0],
# [0,0,0],
# [0,0,0],
# [0,0,0],
# [111,74,0],
# [81,0,81],
# [128,64,128],
# [244,35,232],
# [250,170,160],
# [230,150,140],
# [70,70,70],
# [102,102,156],
# [190,153,153],
# [180,165,180],
# [150,100,100],
# [150,120,90],
# [153,153,153],
# [153,153,153],
# [250,170,30],
# [220,220,0],
# [107,142,35],
# [152,251,152],
# [70,130,180],
# [220,20,60],
# [255,0,0],
# [0,0,142],
# [0,0,70],
# [0,60,100],
# [0,0,90],
# [0,0,110],
# [0,80,100],
# [0,0,230],
# [119,11,32],
# [0,0,142]]
tmp=[[41,169,226],[246,41,132],[228,193,110],[152,16,60],[58,221,254],[155,155,155]
]
###remove repeated colors
givin_colors=[]
for c in tmp:
    if not c in givin_colors:
        givin_colors.append(c)
        #print(c)

givin_colors=np.array(givin_colors)
givin_colors=np.fliplr(givin_colors)
#print(givin_colors.shape)


def PlotText(mask_,target_names_list):
    unq=np.unique(mask_).tolist()[1:]
    print(np.unique(mask_).tolist())
    text_pos={}
    #print('***********')
    for f in unq:
        thresh=0*mask_
        thresh[np.where(mask_==f)]=255

        # You need to choose 4 or 8 for connectivity type
        connectivity = 8
        # Perform the operation to get information about regoins!!!
        #try:
        if len(thresh.shape)>2:
                 thresh=thresh[:,:,0]
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
       # except Exception as e:
       #     st.write(e)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix

        labels = output[1]
        # The third cell is the stat matrix

        #print(np.max(labels))


        radius = 20

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2



        stats = output[2]

        #print(target_names_list[f])
        #print(stats)


        # The fourth cell is the centroid matrix
        centroids = output[3]



        im=cv2.merge((thresh,thresh,thresh))
        #print(im.shape)
        Flag=False

        print('f=',f)
        print(target_names_list)
        current_class=target_names_list[f]

        text_pos[current_class]=[]


        for i in range(1,stats.shape[0]):
            if stats[i][4]>500: #number of pixels bigger than 500 pixels

                #im=cv2.rectangle(im, (stats[i][0], stats[i][1]), (stats[i][0]+stats[i][2], stats[i][1]+stats[i][3]), (0, 255, 0), 1)
                Flag=True
                #print(centroids[i])
                x,y=centroids[i]
                x,y=int(x),int(y)

                text_pos[current_class].append((x,y))
                #print('***********************')



                # Using cv2.circle() method
                # Draw a circle with blue line borders of thickness of 2 px
                #im = cv2.circle(im,(x,y) , radius, color, thickness)



        if Flag==True:
            #print(f,target_names_list[f])
            #plt.imshow(im)
            #plt.show()
            Flag=False

    return text_pos


def AddTextToMask(mask,target_names):

    shape=mask.shape

    text_pos=PlotText(mask,target_names)
    colored_img=givin_colors[mask.reshape(-1)]

    colored_img=colored_img.reshape((shape[0],shape[1],3))

    colored_img=np.array(colored_img,dtype=np.uint8)

    pil_im = Image.fromarray(colored_img)

    for k in text_pos:
         if len(text_pos[k])>0:
                #print(k,len(text_pos[k]))

                text=k

                draw = ImageDraw.Draw(pil_im)
                # use a truetype font
                font = ImageFont.truetype("Aaron-BoldItalic.ttf", 10)

                coords=text_pos[k]
                # Draw the text
                for c in coords:
                    x,y=c
                    draw.text((x,y), text, font=font,fill=(255,255,255,0))

    new_mask=np.array(pil_im)
    return new_mask

import os
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import PIL
import random
from scipy import ndimage


class segDataset(torch.utils.data.Dataset):
    def __init__(self, root, training, transform=None):
        super(segDataset, self).__init__()
        self.root = root
        self.training = training
        self.transform = transform
        self.IMG_NAMES = sorted(glob(self.root + '/*/images/*.jpg'))
        self.BGR_classes = {'Water' : [ 41, 169, 226],
                            'Land' : [246,  41, 132],
                            'Road' : [228, 193, 110],
                            'Building' : [152,  16,  60],
                            'Vegetation' : [ 58, 221, 254],
                            'Unlabeled' : [155, 155, 155]} # in BGR

        self.bin_classes = ['Water', 'Land', 'Road', 'Building', 'Vegetation', 'Unlabeled']


    def __getitem__(self, idx):
        img_path = self.IMG_NAMES[idx]
        mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        cls_mask = np.zeros(mask.shape)
        cls_mask[mask == self.BGR_classes['Water']] = self.bin_classes.index('Water')

        cls_mask[mask == self.BGR_classes['Land']] = self.bin_classes.index('Land')
        cls_mask[mask == self.BGR_classes['Road']] = self.bin_classes.index('Road')
        cls_mask[mask == self.BGR_classes['Building']] = self.bin_classes.index('Building')
        cls_mask[mask == self.BGR_classes['Vegetation']] = self.bin_classes.index('Vegetation')
        cls_mask[mask == self.BGR_classes['Unlabeled']] = self.bin_classes.index('Unlabeled')
        cls_mask = cls_mask[:,:,0]

        if self.training==True:
            if self.transform:
              image = transforms.functional.to_pil_image(image)
              image = self.transform(image)
              image = np.array(image)

            # 90 degree rotation
            if np.random.rand()<0.5:
              angle = np.random.randint(4) * 90
              image = ndimage.rotate(image,angle,reshape=True)
              cls_mask = ndimage.rotate(cls_mask,angle,reshape=True)

            # vertical flip
            if np.random.rand()<0.5:
              image = np.flip(image, 0)
              cls_mask = np.flip(cls_mask, 0)

            # horizonal flip
            if np.random.rand()<0.5:
              image = np.flip(image, 1)
              cls_mask = np.flip(cls_mask, 1)

        image = cv2.resize(image, (512,512))/255.0
        cls_mask = cv2.resize(cls_mask, (512,512))
        image = np.moveaxis(image, -1, 0)

        return torch.tensor(image).float(), torch.tensor(cls_mask, dtype=torch.int64)


    def __len__(self):
        return len(self.IMG_NAMES)

color_shift = transforms.ColorJitter(.1,.1,.1,.1)
blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

aerial_transforms= transforms.Compose([color_shift, blurriness])


 ###******************************************
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from osgeo import gdal

"""
Function which returns the labelled image after applying CRF

"""

#Original_image = Image which has to labelled
#Annotated image = Which has been labelled by some technique( FCN in this case)
#Output_image = Name of the final output image after applying CRF
#Use_2d = boolean variable
#if use_2d = True specialised 2D fucntions will be applied
#else Generic functions will be applied

def crf(original_image, annotated_image,use_2d = True):

    # Converting annotated image to RGB if it is Gray scale
    if(len(annotated_image.shape)<3):
        annotated_image = gray2rgb(annotated_image).astype(np.uint32)

    #cv2.imwrite("testing2.png",annotated_image)
    annotated_image = annotated_image.astype(np.uint32)
    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,0].astype(np.uint32) + (annotated_image[:,:,1]<<8).astype(np.uint32) + (annotated_image[:,:,2]<<16).astype(np.uint32)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat))

    print("No of labels in the Image are ")
    print(n_labels)


    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.90, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    #Run Inference for 5 steps
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    #cv2.imwrite(output_image,MAP.reshape(original_image.shape))
    return MAP.reshape(original_image.shape)
