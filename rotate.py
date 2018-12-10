import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt




def main():
    dirname='C:/Users/oeunju/PycharmProjects/dataset/phd08_png_results/thinning'

    filenames =os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        print(filename)
        im1 = Image.open(full_filename)
        spin_baby = im1.rotate(15, fillcolor='white')
        img_numpy = np.array(spin_baby,'uint8')
        cv2.imwrite('C:/Users/oeunju/PycharmProjects/dataset/phd08_png_results/rotate30/' + filename, img_numpy)

        spin_im2 = im1.rotate(-15, fillcolor='white')
        img_numpy1 = np.array(spin_im2,'uint8')
        cv2.imwrite('C:/Users/oeunju/PycharmProjects/dataset/phd08_png_results/rotate-30/'+ filename, img_numpy1)



if __name__ == '__main__':
    main()