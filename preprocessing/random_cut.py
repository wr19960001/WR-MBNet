"""
裁剪图像至指定的height和width
"""

import cv2
import os
import time
# from tqdm import tqdm
import random
from config import _C as config

size = 512


def random_crop():
    g_count = 1
    image_path = '/home/user/tf_serving_file/hed_data/train/image'
    label_path = '/home/user/tf_serving_file/hed_data/train/label'

    image_num_cut = 20000

    big_image_num = len(os.listdir(image_path))

    image_each = image_num_cut // big_image_num

    if not os.path.exists('/home/user/tf_serving_file/hed_data/train/cut_images'):
        os.mkdir('/home/user/tf_serving_file/hed_data/train/cut_images')
    if not os.path.exists('/home/user/tf_serving_file/hed_data/train/cut_labels'):
        os.mkdir('/home/user/tf_serving_file/hed_data/train/cut_labels')

    start_time = time.time()
    print('Start cutting......')

    image_name_list = os.listdir(image_path)
    label_path_list = os.listdir(label_path)
    image_name_list.sort(key=lambda x: int(str(x).split('.')[0][0]))
    label_path_list.sort(key=lambda x: int(str(x).split('.')[0][0]))
    for image_name in image_name_list:
        print('now is image {}'.format(image_name.split('.')[0]))
        count = 0
        # image = cv2.imread(image_path + '/' + os.listdir(image_path)[i])
        image = cv2.imread(image_path + '/' + image_name)
        # label = cv2.imread(label_path + '/' + os.listdir(label_path)[i], cv2.IMREAD_GRAYSCALE)
        # label = cv2.imread(label_path + '/' + image_name.split('.')[0] + '_label' + '.png')
        label = cv2.imread(label_path + '/' + image_name)
        x_height, x_width = image.shape[0], image.shape[1]
        while count < image_each:
            random_width = random.randint(0, x_width - size - 1)
            random_height = random.randint(0, x_height - size - 1)

            image_roi = image[random_height: random_height + size, random_width: random_width + size, :]

            label_roi = label[random_height: random_height + size, random_width: random_width + size]

            cv2.imwrite(('/home/user/tf_serving_file/hed_data/train/cut_images' + '/%05d.png' % g_count), image_roi)
            cv2.imwrite(('/home/user/tf_serving_file/hed_data/train/cut_labels' + '/%05d.png' % g_count), label_roi)

            count += 1
            g_count += 1

    end_time = time.time()
    print('Finish cutting!  Cost %.2f Seconds.' % (end_time-start_time))


if __name__ == '__main__':
    random_crop()