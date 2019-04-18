import sys
sys.path.append('.')
sys.path.append('D:\\Program Files\\caffe_windows_exe_cup_only_x64')
import caffe
import cv2
import numpy as np
import os
from math import *

deploy = 'models/bbox_regression.prototxt'
caffemodel = 'models/bbox_regression.caffemodel'
net = caffe.Net(deploy, caffemodel, caffe.TEST)

# caffe.set_device(0)
caffe.set_mode_cpu()


def preprocess(img, size, channel_mean=[104, 117, 123]):
    img_new = np.asarray(img, dtype=float)
    # print size
    img_new = cv2.resize(img_new, size)
    for i in range(img_new.shape[-1]):
        img_new[:, :, i] = img_new[:, :, i] - channel_mean[i]
    img_new = np.transpose(img_new, (2, 0, 1))
    return img_new


def bboxRegression(img, bboxes):
    origin_h, origin_w, ch = img.shape

    net.blobs['data'].reshape(len(bboxes), 3, 106, 106)
    idx = 0
    for bbox in bboxes:
        crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        net.blobs['data'].data[idx] = preprocess(crop_img, (106, 106))
        idx += 1
    out = net.forward()['ip2']

    for i in range(0, len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
                regressor[j] += 1
                regressor[j] = log(regressor[j]) * 0.2
            else:
                regressor[j] = 1 - regressor[j]
                regressor[j] = -log(regressor[j]) * 0.2
        bboxes[i, 0] = int(round(max(0, x1 + w * regressor[0])))
        bboxes[i, 1] = int(round(max(0, y1 + h * regressor[1])))
        bboxes[i, 2] = int(round(min(origin_w, x2 + w * regressor[2])))
        bboxes[i, 3] = int(round(min(origin_h, y2 + h * regressor[3])))
        print(regressor)

    return bboxes


img_root = 'images'
infos = [line.strip('\n') for line in open('vision_format_info.txt')]
for info in infos:
    info_strlist = info.split(', ')
    img_name = os.path.join(img_root, info_strlist[0])
    if not 'graduation_photo' in img_name:
        continue
    img = cv2.imread(img_name)
    if img is None:
        continue

    bboxes = np.array(info_strlist[1:], dtype=np.float).reshape(-1, 4)
    bboxes_new = bboxRegression(img, np.copy(bboxes))

    draw = img.copy()
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i, :]
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        # cv2.putText(draw, str('%.6f' % rectangle[-1]), (int(rectangle[0]), int(
        #     rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.rectangle(draw, (int(bboxes[i, 0]), int(bboxes[i, 1])), (int(
            bboxes[i, 2]), int(bboxes[i, 3])), (255, 0, 0), 1)
        cv2.rectangle(draw, (int(bboxes_new[i, 0]), int(bboxes_new[i, 1])), (int(
            bboxes_new[i, 2]), int(bboxes_new[i, 3])), (0, 0, 255), 2)
        # for j in range(0, 106):
        #     cv2.circle(draw, (int(key_pts[i, 2 * j] * w / float(106) + x1),
        #                       int(key_pts[i, 2 * j + 1] * h / float(106) + y1)), 1, (0, 255, 0), 1)
    cv2.imshow("result", draw)
    cv2.waitKey(0)
    # cv2.imwrite('result.jpg', draw)
