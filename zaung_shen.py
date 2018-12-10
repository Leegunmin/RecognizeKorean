import cv2
import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter





# condition1dls pixel 검은색 확인
def pixel_is_black(arr,x,y):
    if arr[x,y] ==1:
        return True
    return False

#condtion2 2개에서 6개의 검은 픽셀 가짐?
def pixel_has_2_to_6_black_neighbors(arr,x,y):
    if(2<=arr[x, y-1] + arr[x+1, y-1] + arr[x+1, y] + arr[x+1, y+1] +
        arr[x, y+1] + arr[x-1, y+1] + arr[x-1, y] + arr[x-1, y-1] <= 6):
        return True
    return False
#condition3 transition확인
def pixel_has_1_white_to_black_neighbor_transition(arr,x,y):
    neighbors = [arr[x, y - 1], arr[x + 1, y - 1], arr[x + 1, y], arr[x + 1, y + 1],
                 arr[x, y + 1], arr[x, y + 1], arr[x - 1, y], arr[x - 1, y - 1],
                 arr[x, y - 1]]
    transitions = sum((a, b) == (0, 1) for a, b in zip(neighbors, neighbors[1:]))
    if transitions == 1:
        return True
    return False

#condition4 p2,
def at_least_one_of_P2_P4_P6_is_white(arr, x, y):
    if (arr[x, y - 1] and arr[x + 1, y] and arr[x, y + 1]) == False:
        return True
    return False
#condition5
def at_least_one_of_P4_P6_P8_is_white(arr, x, y):
    if (arr[x + 1, y] and arr[x, y + 1] and arr[x - 1, y]) == False:
        return True
    return False
#condition4 for step two
def at_least_one_of_P2_P4_P8_is_white(arr, x, y):
    if (arr[x, y - 1] and arr[x + 1, y] and arr[x - 1, y]) == False:
        return True
    return False
def at_least_one_of_P2_P6_P8_is_white(arr, x, y):
    if (arr[x, y - 1] and arr[x, y + 1] and arr[x - 1, y]) == False:
        return True
    return False


def main():
    dirname = 'C:/Users/oeunju/Downloads/1500-1700'
    filenames = os.listdir(dirname)
    for i in range(486,1000, 1):
        dirname2= dirname +'/'+ str(i)

        if not os.path.exists(dirname2):
            exit()
        filenames =os.listdir(dirname2)
        for filename in filenames:
                full_filename =os.path.join(dirname2, filename)
                print(filename)
                img = cv2.imread(full_filename, 0)
                retval, orig_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
                bin_thresh = (orig_thresh == 0).astype(int)
                thinned_thresh = bin_thresh.copy()
                while 1:
                    # make a copy of the thinned threshold array to check for changes
                    thresh_copy = thinned_thresh.copy()
                    # step one
                    pixels_meeting_criteria = []
                    # check all pixels except for border and corner pixels
                    # if a pixel meets all criteria, add it to pixels_meeting_criteria list
                    for i in range(1, thinned_thresh.shape[0] - 1):
                        for j in range(1, thinned_thresh.shape[1] - 1):
                            if (pixel_is_black(thinned_thresh, i, j) and
                                    pixel_has_2_to_6_black_neighbors(thinned_thresh, i, j) and
                                    pixel_has_1_white_to_black_neighbor_transition(thinned_thresh, i, j) and
                                    at_least_one_of_P2_P4_P6_is_white(thinned_thresh, i, j) and
                                    at_least_one_of_P4_P6_P8_is_white(thinned_thresh, i, j)):
                                pixels_meeting_criteria.append((i, j))

                    # change noted pixels in thinned threshold array to 0 (white)
                    for pixel in pixels_meeting_criteria:
                        thinned_thresh[pixel] = 0

                    # step two
                    pixels_meeting_criteria = []
                    # check all pixels except for border and corner pixels
                    # if a pixel meets all criteria, add it to pixels_meeting_criteria list
                    for i in range(1, thinned_thresh.shape[0] - 1):
                        for j in range(1, thinned_thresh.shape[1] - 1):
                            if (pixel_is_black(thinned_thresh, i, j) and
                                    pixel_has_2_to_6_black_neighbors(thinned_thresh, i, j) and
                                    pixel_has_1_white_to_black_neighbor_transition(thinned_thresh, i, j) and
                                    at_least_one_of_P2_P4_P8_is_white(thinned_thresh, i, j) and
                                    at_least_one_of_P2_P6_P8_is_white(thinned_thresh, i, j)):
                                pixels_meeting_criteria.append((i, j))

                    # change noted pixels in thinned threshold array to 0 (white)
                    for pixel in pixels_meeting_criteria:
                        thinned_thresh[pixel] = 0

                    # if the latest iteration didn't make any difference, exit loop
                    if np.all(thresh_copy == thinned_thresh) == True:
                        break

                # convert all ones (black pixels) to zeroes, and all zeroes (white pixels) to ones
                thresh = (thinned_thresh == 0).astype(np.uint8)
                # convert ones to 255 (white)
                thresh *= 255
                dirname_simple = dirname2[-3:]

                # display original and thinned images
                # cv2.imshow('original image', orig_thresh)
                # cv2.imshow('thinned image', thresh)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite('C:/Users/oeunju/Desktop/1500-1700/'+dirname_simple+filename, thresh)





if __name__ == '__main__':
    main()