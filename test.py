# coding=utf-8
from evaluation import *
import sys
import argparse


cfg = {'file_dir': './',
       'overlapRatio': 0.5,  # iou between predicted bounding box and ground truth bounding box
       'cls': 2,  # background id included
       'precision': False,  # calculate precision with 'threshold' or not
       'recall': False,  # calculate precision with 'threshold' or not
       'threshold': 0.5,  # confidence threshold used in calculating precision and recall
       'FPPIW': False,  # FPPI: false positive per image; FPPW: false positive per window(bounding box)
       'roc': False,  # draw roc curve or not
       'pr': False  # draw pr curve or not
        }


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detection Results Test!')
    parser.add_argument('-dir', dest='dir', nargs='+', help='Two folders with detection results and ground truth in '
                                                            'each of them， put detection path in front', type=str)
    parser.add_argument('-ratio', dest='overlapRatio', help='Should be in [0, 1], float type, which means the IOU '
                                                            'threshold, default = 0.5', default=0.5, type=float)
    parser.add_argument('-thre', dest='threshold', help='Should be in [0, 1], float type, if you need [precision] '
                                                        ', [recall], [FPPI] or [FPPW], default = 0.7', default=0.6, type=float)
    parser.add_argument('-cls', dest='cls', help='Should be > 1, which means number of categories(background included),'
                                                     'default = 2', default=2, type=int)
    parser.add_argument('-prec', dest='precision', help='Should be True or False, which means return precision or not, '
                                                        'default = True', default=True, type=bool)
    parser.add_argument('-rec', dest='recall', help='Should be True or False, which means return recall or not, '
                                                    'default = True', default=True, type=bool)
    parser.add_argument('-FPPIW', dest='FPPIW', help='Should be True or False, which means return FPPI and FPPW or not,'
                                                    'default = True', default=True, type=bool)
    parser.add_argument('-roc', dest='roc', help='Should be True or False, which means drawing ROC curve or not, '
                                                    'default = True', default=True, type=bool)
    parser.add_argument('-pr', dest='pr', help='Should be True or False, which means drawing PR curve or not, '
                                                    'default = True', default=True, type=bool)
    args_in = parser.parse_args()

    return args_in


if __name__ == "__main__":
    args = parse_args()
    args.dir = ['/data/guanlang/video_clips/metric_used_xml/face_pred_txt',  # prediction path
                '/data/guanlang/video_clips/metric_used_xml/gt']  # gt path

    # args.cls = 2
    # args.overlapRatio = 0.3
    # args.threshold = 0.94
    len(sys.argv)
    print("Your Folder's path: {}".format(args.dir))
    print("Overlap Ratio: {}".format(args.overlapRatio))
    print("Threshold: {}".format(args.threshold))
    print("Num of Categories: {}".format(args.cls))
    print("Precision: {}".format(args.precision))
    print("Recall: {}".format(args.recall))
    print("FPPIW: {}".format(args.FPPIW))

    print("Calculating......")

    cfg['file_dir'] = args.dir
    cfg['overlapRatio'] = args.overlapRatio
    cfg['cls'] = args.cls
    cfg['precision'] = args.precision
    cfg['recall'] = args.recall
    cfg['threshold'] = args.threshold
    cfg['FPPIW'] = args.FPPIW
    cfg['roc'] = args.roc
    cfg['pr'] = args.pr

    eval = evaluation(cfg)
    eval.run()