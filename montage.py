import matplotlib.pyplot as plt 
from skimage.util import montage as m

def montage(input_arr):
    output_arr = m(input_arr)
    plt.ion()
    plt.figure()
    if len(input_arr.shape) == 3:
        plt.imshow(output_arr, cmap='gray')
        plt.show()
    else:
        plt.imshow(output_arr)
        plt.show()