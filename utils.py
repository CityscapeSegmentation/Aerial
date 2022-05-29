


from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
import streamlit as st

import numpy as np
import cv2
from PIL import ImageDraw,ImageFont,Image



    
trans = transforms.Compose([
    transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
])    

target_names=['avoid','ground','road','sidewalk','parking','building','fence or guard rail',
             'traffic sign or pole','vegetation','terrain','sky','person','vehicle','bike','wall']


class CityDataset(Dataset):
    def __init__(self, data_path,  transform=None):
        self.input_images=os.listdir(data_path+'/rgb')        
        self.transform = transform
        self.data_path=data_path
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):     
        
        data_path=self.data_path
        
        image = np.array(Image.open( data_path+'/rgb/'+ self.input_images[idx])  )
        mask = np.array(Image.open( data_path+'/mask/'+ self.input_images[idx]))
        
        if self.transform:
            image = self.transform(image)
            mask=torch.from_numpy(mask).long()
            #mask=mask.long()
        
        return [image, mask]
    
#val_set = CityDataset(data_path='data/val', transform = trans)
#DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)

#image = np.array(Image.open( data_path+'/rgb/'+ self.input_images[idx])  )

tmp=[
[0,0,0],
[0,0,0],
[0,0,0],
[0,0,0],
[0,0,0],
[111,74,0],
[81,0,81],
[128,64,128],
[244,35,232],
[250,170,160],
[230,150,140],
[70,70,70],
[102,102,156],
[190,153,153],
[180,165,180],
[150,100,100],
[150,120,90],
[153,153,153],
[153,153,153],
[250,170,30],
[220,220,0],
[107,142,35],
[152,251,152],
[70,130,180],
[220,20,60],
[255,0,0],
[0,0,142],
[0,0,70],
[0,60,100],
[0,0,90],
[0,0,110],
[0,80,100],
[0,0,230],
[119,11,32],
[0,0,142]]

###remove repeated colors
givin_colors=[]
for c in tmp:
    if not c in givin_colors:
        givin_colors.append(c)
        #print(c)

givin_colors=np.array(givin_colors)
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
        try:
            output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        except Exception as e:
            st.write(e)
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
    
 
