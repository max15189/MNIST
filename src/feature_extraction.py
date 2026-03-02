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

def intensity_projection(image):
    '''
    input: 1D array of 784 pixel values (28x28)
    output: 1D array of 56 values (28 row sums + 28 column sums)
    the logic is that there are distinct patterns in the way we write digits
    so for example the digit 1 will have a high value in the middle but low at the sides
    '''
    image_2d = image.reshape(28, 28)
    
    row_projection = image_2d.sum(axis=1)  # 28 values
    col_projection = image_2d.sum(axis=0)  # 28 values
    
    return np.concatenate([row_projection, col_projection])
