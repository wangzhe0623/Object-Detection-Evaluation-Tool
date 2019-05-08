# coding=utf-8
from evaluation import *
import sys
import argparse


cfg = {'file_dir': './',
       'overlapRatio': 0.5,
       'cls': 2,
       'presicion': False,
       'recall': False,
       'threshold': 0.5,
       'FPPIW': False,
       'roc': False,
       'pr': False}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detection Results Test!')
    parser.add_argument('-dir', dest='dir', nargs='+', help='Two folders with detection results and ground truth in '
                                                            'each of them， put detection path in front', type=str)
    parser.add_argument('-ratio', dest='overlapRatio', help='Should be in [0, 1], float type, which means the IOU '
                                                            'threshold, default = 0.5', default=0.5, type=float)
    parser.add_argument('-thre', dest='threshold', help='Should be in [0, 1], float type, if you need [precision] '
                                                        ', [recall], [FPPI] or [FPPW], default = 0.5', default=0.5, type=float)
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
    # args.dir = ['/Users/wangzhe/data/safe_belt/part1/prediction/', '/Users/wangzhe/data/safe_belt/part1/test_annos/']
    # args.cls = 4
    len(sys.argv)
    print ("Your Folder's path: {}".format(args.dir))
    print ("Overlap Ratio: {}".format(args.overlapRatio))
    print ("Threshold: {}".format(args.threshold))
    print ("Num of Categories: {}".format(args.cls))
    print ("Precision: {}".format(args.precision))
    print ("Recall: {}".format(args.recall))
    print ("FPPIW: {}".format(args.FPPIW))

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
