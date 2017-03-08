import os

import tensorflow as tf
import pandas as pd
from more_itertools import unique_everseen

import voc_utils as voc

def load_train_data(category):
    to_find = category
    train_filename = 'TrainVal/VOCdevkit/VOC2011/ImageSets/Main/train_' + category + '.csv'
    if os.path.isfile(train_filename):
        return pd.read_csv(train_filename)
    else:
        train_img_list = voc.imgs_from_category_as_list(to_find, 'train')
        data = []
        for item in train_img_list:
            anno = voc.load_annotation(item)
            objs = anno.findAll('object')
            for obj in objs:
                obj_names = obj.findChildren('name')
                for name_tag in obj_names:
                    if str(name_tag.contents[0]) == 'bicycle':
                        fname = anno.findChild('filename').contents[0]
                        bbox = obj.findChildren('bndbox')[0]
                        xmin = int(bbox.findChildren('xmin')[0].contents[0])
                        ymin = int(bbox.findChildren('ymin')[0].contents[0])
                        xmax = int(bbox.findChildren('xmax')[0].contents[0])
                        ymax = int(bbox.findChildren('ymax')[0].contents[0])
                        data.append([fname, xmin, ymin, xmax, ymax])
        df = pd.DataFrame(data, columns=['fname', 'xmin', 'ymin', 'xmax', 'ymax'])
        df.to_csv(train_filename, index=False)
        return df


if __name__ == "__main__":
    train_img_list = voc.imgs_from_category_as_list('bicycle', 'train')
    a = voc.load_annotation(train_img_list[0])
    df = load_train_data('bicycle')
    print(list(unique_everseen(list(voc.img_dir + df['fname']))))
