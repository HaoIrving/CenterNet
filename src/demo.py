from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import json
import pandas as pd

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

#        results = {}
        im_id = []
        im_h = []
        im_w = []
        string = []
        for (image_name) in image_names:
            ret, info = detector.run(image_name)
            image = cv2.imread(image_name)
            h, w = image.shape[0:2]
            save_name = image_name.split('/')[-1][:-4] + '.xml'
            #results[save_name] = info
            row = []
            for pred in info:
                b_norm = [pred[0][0]/w, pred[0][1]/h, pred[0][2]/w, pred[0][3]/h]
                b_str = [str(m) for m in b_norm]
                box_str = " ".join(b_str)
                obj = pred[1]+' '+str(pred[2])+' '+box_str
                row.append(obj)
            pred_str = " ".join(row)
            string.append(pred_str)
            im_h.append(h)
            im_w.append(w)
            im_id.append(save_name)

            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
        res = pd.DataFrame(im_id, columns=['image_id']) #results
        res['PredictionString'] = string
        res['im_h'] = im_h
        res['im_w'] = im_w
        csv_name = opt.save_dir+"/result_{}_{}_{}.csv".format('test-A', 'flip_multi_scale', opt.exp_id)
        res.to_csv(csv_name, index=False)
'''
        results_str = json.dumps(results)
        if opt.flip_test == True and len(opt.test_scales) is not 1:
          with open(opt.save_dir+"/result_{}_{}_{}.json".format('test-A', 'flip_multi_scale', opt.exp_id), 'w') as json_file:
            json_file.write(results_str)
          print('result of flip + multi scale augmentation, saved in ', opt.save_dir)
        elif opt.flip_test == True and len(opt.test_scales) is 1:
          with open(opt.save_dir+"/result_{}_{}.json".format('test-A', 'flip'), 'w') as json_file:
            json_file.write(results_str)
          print('result of flip augmentation, saved in ', opt.save_dir)
        elif opt.flip_test == False and len(opt.test_scales) is not 1: #后面L237改成了list
          with open(opt.save_dir+"/result_{}_{}.json".format('test-A', 'multi_scale'), 'w') as json_file:
            json_file.write(results_str)
          print('result of multi scale test augmentation, saved in ', opt.save_dir)
        else:
          with open(opt.save_dir+"/result_{}.json".format('test-A'), 'w') as json_file:
            json_file.write(results_str)
          print('result with no augmentation, saved in ', opt.save_dir)
'''


if __name__ == '__main__':
    opt = opts().init()
    demo()
