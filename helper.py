
import numpy as np
from PIL import Image

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image_path) as img:
        #resize the shorestes size
        re_size = 256
        new_width = 224
        new_height = 224
        means = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        width = img.size[0]
        height = img.size[1]
        size_ratio = width / height
        
        min_size = min((width,height))
        max_size = max((width,height))

        #resize image
        if min_size > 256 or max_size < 256:
            
            if width > height :
                img = img.resize((int(re_size *size_ratio) ,re_size))
            else :
                img = img.resize((re_size,int(re_size / size_ratio)))
        elif min_size < 256 and max_size >256 :
            if width > height :
                img = img.resize((width * re_size // height ,re_size))
            else :
                img = img.resize((re_size,height * re_size // width ))
            
        # crop image
        left = (img.size[0] - new_width)/2
        top = (img.size[1]- new_height)/2
        right = (img.size[0] + new_width)/2
        bottom = (img.size[1] + new_height)/2
        img = img.crop((left, top, right, bottom))        
        img = np.array(img) / 255.0

        # standarized color channels
        for i in range(2):
            img[:,:,i]  = (img[:,:,i] - means[i]) / std[i]
        
        
    return img



