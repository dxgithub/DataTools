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


caffe.set_mode_cpu()
#caffe.set_device(0)

def view_bar(num, total):
    rate = float(num) / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%  (%d/%d)' % ("#" * rate_num, " " *
                                   (100 - rate_num), rate_num, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()



def preprocess(img, size, channel_mean=[104, 117, 123]):
    img_new = np.asarray(img, dtype=float)
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
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        #x2=max(x1+1,x2)
        #y2=max(y1+1,y2)
        if (x1>origin_w or x2>origin_w or y1>origin_h or y2>origin_h or x1<0 or x2<0 or y1<0 or y2<0 or x2<=x1 or y2<=y1):
            continue
        #print("Original image size : "+str(img.shape))
        #print("crop window:"+str([x1,y1,x2,y2]))
        crop_img = img[y1:y2,x1:x2,:]
        #print(str(crop_img.shape))
        net.blobs['data'].data[idx] = preprocess(crop_img, (106, 106))
        idx += 1
    out = net.forward()['ip2']

    for i in range(0, len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        regressor = out[i, :]
        for j in range(len(regressor)):
            if regressor[j] > 0:
                regressor[j] += 1
                regressor[j] = log(regressor[j]) * 0.2
            else:
                regressor[j] = 1 - regressor[j]
                regressor[j] = -log(regressor[j]) * 0.2
        bboxes[i, 0] = int(round(max(0, x1 + w * regressor[0])))
        bboxes[i, 1] = int(round(max(0, y1 + h * regressor[1])))
        bboxes[i, 2] = int(round(min(origin_w, x2 + w * regressor[2])))
        bboxes[i, 3] = int(round(min(origin_h, y2 + h * regressor[3])))
        #print(regressor)

    return bboxes

def writeLabel(filePonter,imageName,Boxes):
    label=[]
    filePonter.write(imageName+" ")
    b=Boxes.reshape(-1).tolist()
    for pos in b:
        label.append(pos)
    filePonter.write(str(label).strip("[]").replace(",",' '))
    filePonter.write("\n")

img_root = 'E:/Data/FaceG/IR_labelled/Training/'
infos = [line.strip('\n') for line in open("E:/Data/FaceG/IR_labelled/IR_Training_label.txt")]
N=len(infos)
view_idx=0
f=open("E:/Data/FaceG/IR_labelled/IR_Training_Regressed_label.txt",'w')
for info in infos:
        view_idx=view_idx+1
        info_strlist = info.split(',')
        img_name =img_root+info_strlist[0]+".jpg"
        #if not 'graduation_photo' in img_name:
        #    continue
        print(img_name)
        img = cv2.imread(img_name)
        if img is None:
            print("NULL Images "+img_name)
            continue
        info_strlist=[xx for xx in info_strlist if xx != ""]
        bboxes = np.array(info_strlist[1:], dtype=np.float).reshape(-1, 4)
        bboxes_new = bboxRegression(img, np.copy(bboxes))
        writeLabel(f,info_strlist[0].strip('\''),bboxes_new)
        view_bar(view_idx, N)
        draw = img.copy()
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i, :]
            # w = x2 - x1 + 1
            # h = y2 - y1 + 1
            #cv2.putText(draw, str('%.6f' % rectangle[-1]), (int(rectangle[0]), int(
            #    rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            #cv2.rectangle(draw, (int(bboxes[i, 0]), int(bboxes[i, 1])), (int(
            #    bboxes[i, 2]), int(bboxes[i, 3])), (255, 0, 0), 1)
            cv2.rectangle(draw, (int(bboxes[i, 0]), int(bboxes[i, 1])), (int(
                bboxes[i, 2]), int(bboxes[i, 3])), (0, 255, 0), 2)
            cv2.rectangle(draw, (int(bboxes_new[i, 0]), int(bboxes_new[i, 1])), (int(
                bboxes_new[i, 2]), int(bboxes_new[i, 3])), (0, 0, 255), 2)
            # for j in range(0, 106):
            #     cv2.circle(draw, (int(key_pts[i, 2 * j] * w / float(106) + x1),
            #                       int(key_pts[i, 2 * j + 1] * h / float(106) + y1)), 1, (0, 255, 0), 1)
        #cv2.imshow("result", draw)
        #cv2.waitKey(0)
        outdir="E:/Data/FaceG/IR_labelled/RegressedImages/"+str(info_strlist[0]+".jpg").strip('\'')
        cv2.imwrite(outdir, draw)
f.close()
