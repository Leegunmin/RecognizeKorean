import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def parse_args():
    desc = "phd08 �ѱ� �ؽ�Ʈ �����͸� .png �������� ��ȯ���ִ� ��ũ��Ʈ�Դϴ�."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_dir', type=str, default='phd08_sample',
                        help='phd08 �ѱ� �����Ͱ� �����ϴ� ���丮', required=True)
    parser.add_argument('--width', type=int, default=40,
                        help='png �� �����Ҷ��� ���� �ȼ�', required=False)
    parser.add_argument('--height', type=int, default=40,
                        help='png �� �����Ҷ��� ���� �ȼ�', required=False)
    parser.add_argument('--gaussian_sigma', type=float, default=.3,
                        help='����þ� ���� ���� �� �ñ׸� ��', required=False)

    return parser.parse_args()


def font_start_checker(line):
    if not line.strip(): #line�� ���� ������ �����.

        return True

    else:

        return False



def txt_to_png(file_orig_path, file_full_path, width, height, sigma):

    file_orig_path = file_orig_path[:-4]
    save_dir = 'phd08_png_results/' + file_orig_path


    if not os.path.exists('phd08_png_results/' + file_orig_path):
        os.makedirs(save_dir)


    with open(file_full_path, 'r') as lines: #lines�� ����Ʈ�ΰ�
        font_counter = 0
        not_data_checker = 0
        font_array = None
        font_name = ''
        real_data_counter = 0
        for line in lines:
            if font_start_checker(line):  # endl
                # save
                font_blurred_array = gaussian_filter(font_array, sigma=sigma)
                fig = plt.figure(frameon=False)

                fig.set_size_inches(width, height)
                ax = plt.Axes(fig, [0., 0., 1., 1.]) #�ϳ��� �׸�

                ax.set_axis_off() #turn off the axis
                fig.add_axes(ax) #   ax = plt.Axes(fig, [0., 0., 1., 1.])�� �׸��� ��
                ax.imshow(font_blurred_array, aspect='auto', cmap='Greys') #blured array�� grey�� �ٲ���
                fig.savefig('phd08_png_results/' + file_orig_path + '/' + font_name + '.png', dpi=1)
                print('phd08_png_results/' + file_orig_path +'/' + font_name + '.png')
                plt.close(fig)

                not_data_checker = 0
                font_counter += 1
                real_data_counter = 0
                continue
            else:  # not endl
                not_data_checker += 1
                if not_data_checker == 1:  # font name
                    font_name = line.strip()

                    continue
                elif not_data_checker == 2:  # font size
                    arr_height = int(line.split(' ')[0])
                    arr_width = int(line.split(' ')[1])
                    font_array = np.zeros(shape=(arr_height, arr_width))

                    continue
                else:  # this is real data
                    font_array[real_data_counter] = list(map(int, line.strip()))
                    real_data_counter += 1
                    continue


def main():
    args = parse_args()
    if args is None:
        exit()

    # �������� üũ
    if not os.path.exists(args.data_dir):
        print("ERROR::" + args.data_dir, " �� �������� �ʴ� �����Դϴ�.")
        exit()

    for _, _, files in os.walk(args.data_dir):
        for file in files:
            if file[0] == '.':
                continue
            print("INFO:: converting " + file + "...")
            txt_to_png(file, args.data_dir + '/' + file, args.width, args.height, args.gaussian_sigma)

    print("INFO:: all files converted to png, results in phd08_png_results/")


if __name__ == '__main__':
    main()