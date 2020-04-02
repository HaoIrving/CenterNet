import json
import csv

def read_json(path):
    with open(path,'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict
    
path = '/home/gqp/centernet_underwater/CenterNet/exp/ctdet/coco_hg_140epoch/resultsA.json'

dict = read_json(path)

headers = ['name', 'image_id', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']



rows = [
        [1,'xiaoming','male',168,23],
        [1,'xiaohong','female',162,22],
        [2,'xiaozhang','female',163,21],
        [2,'xiaoli','male',158,21]
    ]

with open('test.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)
