import matplotlib.pyplot as plt 
from skimage.util import montage as m

def montage(input):
    output = montage(input)
    if len(output.shape) == 3:
        plt.imshow(output, cmap='gray')
        plt.show()
    else:
        plt.imshow(output)
        plt.show()