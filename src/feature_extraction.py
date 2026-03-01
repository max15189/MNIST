from skimage.feature import hog
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
def extract_hog_features(image,vis_flag=False):
    image_2d = image.reshape(28, 28)
    
    hog_features = hog(
        image_2d,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    
    )
    if vis_flag:
    
        hog_features, hog_image = hog(
        image_2d,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    
        )        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        axes[0].imshow(image_2d, cmap='gray')
        axes[0].set_title('Original')
        
        axes[1].imshow(hog_image, cmap='magma')
        axes[1].set_title('HOG')
        
        plt.tight_layout()
        plt.show()
    
    return hog_features