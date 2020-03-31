from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        results = {}
        for (image_name) in image_names:
            ret, info = detector.run(image_name)
            save_name = image_name.split('/')[-1]
            results[save_name] = info
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)

        results_str = json.dumps(results)
        if opt.flip_test == True and len(opt.test_scales) is not 1:
          with open(opt.save_dir+"/result_{}_{}.json".format('test-A', 'flip + multi scale'), 'w') as json_file:
            json_file.write(results_str)
          print('result of flip + multi scale augmentation, saved in ', opt.save_dir)
        elif opt.flip_test == True and len(opt.test_scales) is 1:
          with open(opt.save_dir+"/result_{}_{}.json".format('test-A', 'flip'), 'w') as json_file:
            json_file.write(results_str)
          print('result of flip augmentation, saved in ', opt.save_dir)
        elif opt.flip_test == False and len(opt.test_scales) is not 1: #后面L237改成了list
          with open(opt.save_dir+"/result_{}_{}.json".format('test-A', 'multi scale test'), 'w') as json_file:
            json_file.write(results_str)
          print('result of multi scale test augmentation, saved in ', opt.save_dir)
        else:
          with open(opt.save_dir+"/result_{}.json".format('test-A'), 'w') as json_file:
            json_file.write(results_str)
          print('result with no augmentation, saved in ', opt.save_dir)


if __name__ == '__main__':
    opt = opts().init()
    demo()
