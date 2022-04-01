"""
重采样不平衡数据
"""

import numpy as np
import cv2
import os
import time


def crop_data(image_path, label_path, output_dir, size):
    if not os.path.exists(os.path.join(output_dir, 'cut_images')):
        print('No images dir exist, create output dir in {} now.'.format(output_dir))
        os.mkdir(os.path.join(output_dir, 'cut_images'))
    if not os.path.exists(os.path.join(output_dir, 'cut_labels')):
        print('No labels dir exist, create output dir in {} now.'.format(output_dir))
        os.mkdir(os.path.join(output_dir, 'cut_labels'))

    cut_image_path = os.path.join(output_dir, 'cut_images')
    cut_label_path = os.path.join(output_dir, 'cut_labels')

    g_count = 1

    start_time = time.time()
    print('Start cutting......')

    image_name_list = os.listdir(image_path)
    label_name_list = os.listdir(label_path)
    image_name_list.sort()
    label_name_list.sort()

    unit_pixel_all = size * size * 3

    for image_name in image_name_list:
        h_index = 0
        print('Now is image {}'.format(image_name.split('.')[0]))

        image = cv2.imread(image_path + '/' + image_name)
        label = cv2.imread(label_path + '/' + 'label.png')
        h, w, _ = image.shape
        while h_index < h - size:
            w_index = 0
            while w_index < w - size:
                image_roi = image[h_index: h_index + size, w_index: w_index + size, :]
                label_roi = label[h_index: h_index + size, w_index: w_index + size]
                cv2.imwrite((cut_image_path + '/%05d.png' % g_count), image_roi)
                cv2.imwrite((cut_label_path + '/%05d.png' % g_count), label_roi)

                g_count += 1

                if len(np.where(label_roi == 0)[0]) / unit_pixel_all <= 0.95:
                    w_index = w_index + size // 2
                else:
                    w_index = w_index + size // 10 * 9

                w_index = w_index + size // 10 * 9

            h_index = h_index + size // 10 * 9

    end_time = time.time()
    print('Finish cutting! Cost %.2f Seconds.' % (end_time - start_time))


if __name__ == '__main__':
    img_path = '/home/user/tf_serving_file/project_data/data/2014/2014_82/2014_82_json/img'
    lab_path = '/home/user/tf_serving_file/project_data/data/2014/2014_82/2014_82_json/label'
    out_dir = '/home/user/tf_serving_file/project_data/data/2014/2014_82/2014_82_json/all'
    crop_data(img_path, lab_path, out_dir, 512)