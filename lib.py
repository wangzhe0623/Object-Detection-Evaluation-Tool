# coding=utf-8
import os
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import utils

def IntersectBBox(bbox1, bbox2):
    intersect_bbox = []
    if bbox2[0] >= bbox1[2] or bbox2[2] <= bbox1[0] or bbox2[1] >= bbox1[3] or bbox2[3] <= bbox1[1]:
        # return [0, 0, 0, 0], if there is no intersection
        return intersect_bbox
    else:
        intersect_bbox.append([max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]),
                               min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])])
    return intersect_bbox


def JaccardOverlap(bbox1, bbox2):
    intersect_bbox = IntersectBBox(bbox1, bbox2)
    if len(intersect_bbox) == 0:
        return 0
    else:
        intersect_width = int(intersect_bbox[0][2]) - int(intersect_bbox[0][0])
        intersect_height = int(intersect_bbox[0][3]) - int(intersect_bbox[0][1])
        if intersect_width and intersect_height > 0:
            intersect_size = float(intersect_width) * float(intersect_height)
            bbox1_size = float(bbox1[3] - bbox1[1]) * float(bbox1[2] - bbox1[0])
            bbox2_size = float(bbox2[3] - bbox2[1]) * float(bbox2[2] - bbox2[0])
            return float(intersect_size / float(bbox1_size + bbox2_size - intersect_size))
        else:
            return 0


def cumTpFp(gtFile, detFile, label, overlapRatio, file_format):
    # gtRect: label, xmin, ymin, xmax, ymax
    gtRects = []
    # gtRect: label, xmin, ymin, xmax, ymax, score
    detRects = []
    # scores: scores for label
    scores = []
    num_pos = 0
    if file_format[0] == '.txt':
        gtCon = open(gtFile)
        gtLines = gtCon.readlines()
        for gtLine in gtLines:
            if gtLine.split(' ')[0] == str(label):
                gtRects.append((int(float(gtLine.split(' ')[1])), int(float(gtLine.split(' ')[2])),
                                int(float(gtLine.split(' ')[3])), int(float(gtLine.split(' ')[4].strip('\n')))))
                num_pos += 1
    elif file_format[0] == '.xml':
        rects_xml = utils.parse_xml(gtFile)
        for rect_xml in rects_xml:
            if rect_xml[0] == str(label):
                gtRects.append((rect_xml[0], rect_xml[1], rect_xml[2], rect_xml[3]))
                num_pos += 1

    detCon = open(detFile)
    detLines = detCon.readlines()
    for detLine in detLines:
        if detLine.split(' ')[0] == str(label):
            detRects.append((int(detLine.split(' ')[1]), int(detLine.split(' ')[2]),
                             int(detLine.split(' ')[3]), int(detLine.split(' ')[4])))
            scores.append(float(detLine.split(' ')[5].strip('\n')))
    # det_state: [label, score, tp, fp], tp, fp = 0 or 1
    det_state = [(label, 0., 0, 1)] * len(detRects)
    iou_max = 0
    maxIndex = -1
    blockIdx = -1
    for cnt in range(len(det_state)):
        det_state[cnt] = (label, scores[cnt], 0, 1)
    visited = [0] * len(gtLines)
    if len(detRects) != len(scores):
        print("Num of scores does not match detection results!")
    for indexDet, deti in enumerate(detRects):
        iou_max = 0
        maxIndex = -1
        blockIdx = -1
        for indexGt, gti in enumerate(gtRects):
            iou = JaccardOverlap(detRects[indexDet], gtRects[indexGt])
            if iou > iou_max:
                iou_max = iou
                maxIndex = indexDet
                blockIdx = indexGt
        if iou_max >= overlapRatio and visited[blockIdx] == 0:
            det_state[maxIndex] = (label, scores[indexDet], 1, 0)
            visited[blockIdx] = 1

    return det_state, num_pos


def CumSum(tp):
    # tp_copy = tp
    tp_copy = sorted(tp, key=itemgetter(0), reverse=True)
    cumsum = []
    cumPre = 0
    # tp_th = 0
    for index, pair in enumerate(tp_copy):
        cumPre += (tp_copy[index][1])
        cumsum.append(cumPre)

    return cumsum


def CumSum_tp(tp, threshold):
    # tp_copy = tp
    tp_copy = sorted(tp, key=itemgetter(0), reverse=True)
    cumsum = []
    cumPre = 0
    tp_th = 0
    tp_th_num = 0
    for index, pair in enumerate(tp_copy):
        cumPre += (tp_copy[index][1])
        cumsum.append(cumPre)
        if tp_copy[index][0] > threshold:
            tp_th_num += 1
            if tp_copy[index][1] == 1:
                tp_th += 1
    tp_precision = float(tp_th) / float(tp_th_num)
    return cumsum, tp_th, tp_precision


def computeAp(tp, all_num_pos, fp, threshold, label):
    num = len(tp)
    prec = []
    rec = []
    fpr = []
    ap = 0
    if num == 0 or all_num_pos == 0:
        return prec, rec, ap
    tp_cumsum = []
    tp_cumsum, tp_th, tp_precision = CumSum_tp(tp, threshold)
    fp_cumsum = []
    fp_cumsum = CumSum(fp)

    # Compute precision. Compute recall.
    # precTXT = open('prec.txt', 'w')
    # recTXT = open('rec.txt', 'w')
    for i in range(num):
        prec.append(float(tp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]))
        rec.append(float(tp_cumsum[i]) / float(all_num_pos))
        fpr.append(float(fp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]))
        # precTXT.write(str(float(tp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i])) + "\n")
        # recTXT.write(str(float(tp_cumsum[i]) / float(all_num_pos)) + '\n')
        # if float(tp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]) < 1:
        #     print("prec: ", i, ': ', float(tp_cumsum[i]) / float(tp_cumsum[i] + fp_cumsum[i]))
        
    # 画roc曲线图
    plt.figure('Draw')
    plt.plot(fpr, rec)  # plot绘制折线图
    # x_y_ticks = []
    # for i in range(0, 1, float(0.001)):
    #     x_y_ticks.append(i)
    # plot.
    plt.grid(True)
    plt.xlabel('fp')
    plt.ylabel('rc')
    plt.draw()  # 显示绘图
    # plt.pause(5)  # 显示5秒
    plt.savefig("{}.jpg".format(label))  # 保存图象

    plt.close()
    tmp = 0
    # VOC2007 style for computing AP.
    max_precs = [0.] * 11
    start_idx = num - 1
    j = 10
    while j >= 0:
        i = start_idx
        while i >= 0:
            tmp = j / 10.0
            if rec[i] < tmp:
                start_idx = i
                if j > 0:
                    max_precs[j - 1] = max_precs[j]
                    break
            else:
                if max_precs[j] < prec[i]:
                    max_precs[j] = prec[i]
            i -= 1
        j -= 1
    for iji in range(11):
        ap += max_precs[iji] / 11.0

    # 计算 recall 和 precision
    recall = float(tp_th) / float(all_num_pos)
    precision = tp_precision
    return precision, recall, ap


def getAp(gt_path, prediction_path, groundtruths, predictions, label, overlapRatio, threshold, file_format):
    state_all = []
    # tp = [(float, int)]
    # fp = [(float, int)]
    tp = []
    fp = []
    all_num_pos = 0
    for groundtruth in groundtruths:
        name = groundtruth.strip(file_format[1])
        prediction = name + file_format[0]
        if prediction not in predictions:
            print(groundtruth, ': can not find corresponding file in prediction!')
            return 0, 0, 0
        groundtruth = os.path.join(gt_path, groundtruth)
        prediction = os.path.join(prediction_path, prediction)

        det_state, num_pos = cumTpFp(groundtruth, prediction, label, overlapRatio, file_format)
        all_num_pos += num_pos
        state_all += det_state

    for state in state_all:
        # print(state_all)
        tp.append((state[1], state[2]))
        fp.append((state[1], state[3]))
        # tpTXT.write(str(state[1]) + "  " + str(state[2]) + "\n")
        # fpTXT.write(str(state[1]) + "  " + str(state[3]) + "\n")

    if len(tp) != len(fp):
        print("tp size != fp size, please check ~")
    precision, recall, ap = computeAp(tp, all_num_pos, fp, threshold, label)

    return precision, recall, ap


def evaluation(path, overlapRatio, classes, precision, recall, threshold):
    prediction_path = path[0]
    gt_path = path[1]
    if not os.path.exists(prediction_path):
        print('Incorrect detection results path! Please check and retry ~')
        return 0
    if not os.path.exists(gt_path):
        print('Incorrect ground truth path! Please check and retry ~')
        return 0
    if overlapRatio < 0 or overlapRatio > 1:
        print('Incorrect overlapRatio value! It should be in [0, 1]. Please check and retry ~')
        return 0
    if threshold < 0 or threshold > 1:
        print('Incorrect threshold value! It should be in [0, 1]. Please check and retry ~')
        return 0
    pre_files = os.listdir(prediction_path)
    print ("Num of prediction files: ", len(pre_files))
    gt_files = os.listdir(gt_path)
    print ("Num of ground truth files: ", len(gt_files))
    if len(pre_files) != len(gt_files):
        print("groundtruths' size does not match predictions' size， please check ~ ")
        return 0
    elif len(pre_files) < 1:
        print('No files! Please check~')
        return 0
    predictions = []
    groundtruths = []
    file_format = ['.txt'] * 2  # [预测的格式， gt的格式]
    for idx, item in enumerate(pre_files):
        if idx == 0:
            if item.find('xml') >= 0:
                file_format[0] = '.xml'
                print('Got prediction file input in XML type ')
            elif item.find('txt'):
                file_format[0] = '.txt'
                print('Got prediction file input in TXT type ')
            else:
                print('Unknown type!')
                return 0
        predictions.append(os.path.join(item))
    for idx, item in enumerate(gt_files):
        if idx == 0:
            if item.find('xml') >= 0:
                file_format[1] = '.xml'
                print('Got ground truth file input in XML type ')
            elif item.find('txt'):
                file_format[1] = '.txt'
                print('Got ground truth file input in TXT type ')
            else:
                print('Unknown type!')
                return 0
        groundtruths.append(os.path.join(item))

    # 循环计算每一类的Ap
    aps = 0
    for label in range(1, classes):
        prec, rec, ap = getAp(gt_path, prediction_path, groundtruths, predictions, label, overlapRatio, threshold, file_format)
        if prec == rec == ap == 0:
            return 0
        print ("class ", label, " Ap: ", ap)

        if not precision:
            print("class ", label, "precision: ", prec)
        if not recall:
            print("class ", label, "recall: ", rec)
        aps += ap
    mAp = aps / (classes - 1)
    print ("mAp: ", mAp)

    return 0
