import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os,cv2

class img_dataset:

    def __init__(self,Shape,Channel):
        self.shape = Shape
        self.channel = Channel


    @staticmethod
    def get_imgInfo(file_dir,have_group = False):

        if have_group :
            Categories = os.listdir(file_dir)
            for category in Categories:
                print('{} {} images'.format(category, len(os.listdir(os.path.join(file_dir,category)))))
            #Load the data to dataframe
            data=[]
            for category_id, category in enumerate(Categories):
                for imgfile in os.listdir(os.path.join(file_dir,category)):
                    data.append(['{}/{}/{}'.format(file_dir,category,imgfile),category_id,category])
            column = ['img_path','category_id','category']
            print("Note --> categories available !")
        else:
            data = []
            for imgfile in os.listdir(file_dir):
                data.append(['{}/{}'.format(file_dir,imgfile),imgfile])
            column = ['img_path','img']
        return pd.DataFrame(data,columns=column)


    def disp_group_image(self,data,no_img=6):
         #if there is 'category' inside the dataframe columns
         #np.any(axis=none) : at least 1 true appear, return true
        if np.any(data.columns == 'category'):
             group = data['category'].unique()
             Num_group = len(group)
             fig = plt.figure(1, figsize=(Num_group,Num_group))
             grid = ImageGrid(fig, 111, nrows_ncols=(Num_group, Num_group), axes_pad=0.05)
             i2 = -1
             for grp in group:
                 img_path = np.array(data[data['category'] == grp]['img_path'])
                 for i,imgdir in enumerate(img_path):
                     if i < Num_group:
                         i2 = i2 + 1
                         img = cv2.imread(imgdir)
                         img = cv2.resize(img,dsize =(self.shape,self.shape),interpolation = cv2.INTER_AREA)
                         ax = grid[i2] #Put img into grid with specific position
                         ax.imshow(img)
                         ax.axis('off')
                     else:
                         #Display each group name across each row
                         ax.text(self.shape+30, self.shape/2, grp, verticalalignment='center')
                         break
        else:
             i2 = -1
             fig = plt.figure(1, figsize=(no_img,no_img))
             grid = ImageGrid(fig, 111, nrows_ncols=(no_img, no_img), axes_pad=0.05)
             img_path = np.array(data['img_path'])
             for i,imgdir in enumerate(img_path):
                if i < no_img*no_img:
                    i2 = i2 + 1
                    img = cv2.imread(imgdir)
                    img = cv2.resize(img,dsize =(self.shape,self.shape),interpolation = cv2.INTER_AREA)
                    ax = grid[i2] #Put img into grid with specific position
                    ax.imshow(img)
                    ax.axis('off')
                else:
                    break
        plt.show()


    def load_image(self,data,extract = False,lwr_hsv=None,upr_hsv=None):
        img_data = []
        img_path = data['img_path']
        getImg = True
        for imgdir in img_path:
            img = cv2.imread(imgdir)

            if extract :
                if lwr_hsv and upr_hsv :
                    blurr = cv2.GaussianBlur(img,(5,5),0)
                    hsv = cv2.cvtColor(blurr,cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv,lwr_hsv,upr_hsv)
                    struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
                    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,struc)
                    boolean = mask>0
                    new_img = np.zeros_like(img,np.uint8)
                    new_img[boolean] = img[boolean]
                    res_img = new_img
                    if getImg:
                        plt.subplot(2,3,1);plt.imshow(img)# ORIGINAL
                        plt.subplot(2,3,2);plt.imshow(blurr) # BLURRED
                        plt.subplot(2,3,3);plt.imshow(hsv) # HSV CONVERTED
                        plt.subplot(2,3,4);plt.imshow(mask) # MASKED
                        plt.subplot(2,3,5);plt.imshow(boolean) # BOOLEAN MASKED
                        plt.subplot(2,3,6);plt.imshow(new_img)# NEW PROCESSED IMAGE
                        plt.axis('off')
                        plt.show()
                        getImg = False
                else:
                    raise ValueError('lwr_hsv or upr_hsv arguments are empty !')

            else:
                blurr = cv2.GaussianBlur(img,(5,5),0)
                rgb = cv2.cvtColor(blurr,cv2.COLOR_BGR2RGB)
                res_img = rgb

            rez_img = cv2.resize(res_img,dsize =(self.shape,self.shape),interpolation = cv2.INTER_AREA)
            rez_img = rez_img.reshape(self.shape,self.shape,self.channel)
            img_data.append(rez_img)

        return img_data

class regression_dataset:
    
    @staticmethod
    def disp_miss_data(data):
        #Display total missing data for each features
        dataNull = data.isnull().sum().sort_values(ascending=False)
        #Calculate the % of missing data for each features
        percent = (data.isnull().sum()/data.shape[0]).sort_values(ascending=False) # / : division for floating point, // ： division for interger point
        #Let above both data series join together
        return  pd.concat([dataNull, percent], axis=1, keys=['Total', 'Percent'])
