# coding=utf-8
import lib
import sys
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Detection Results Test!')
    parser.add_argument('-dir', dest='dir', nargs='+', help='Two folders with detection results and ground truth in '
                                                            'each of them， put detection path in front', type=str)
    parser.add_argument('-ratio', dest='overlapRatio', help='Should be in [0, 1], float type, which means the IOU '
                                                            'threshold, default = 0.5', default=0.5, type=float)
    parser.add_argument('-thre', dest='threshold', help='Should be in [0, 1], float type, if you need [precision] '
                                                        'and [recall], default = 0.5', default=0.5, type=float)
    parser.add_argument('-cls', dest='classes', help='Should be > 1, which means number of categories, default = 2',
                        default=2, type=int)
    parser.add_argument('-prec', dest='precision', help='Should be 0 or 1, which means return precision or not, '
                                                        'default = 0', default=0, type=int)
    parser.add_argument('-rec', dest='recall', help='Should be 0 or 1, which means return recall or not, '
                                                    'default = 0', default=0, type=int)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    len(sys.argv)
    print ("Your Folder's path: {}".format(args.dir))
    print ("Overlap Ratio: {}".format(args.overlapRatio))
    print ("Threshold: {}".format(args.threshold))
    print ("Num of Categories: {}".format(args.classes))
    print ("Precision: {}".format(args.precision))
    print ("Recall: {}".format(args.recall))

    print("Calculating......")

    lib.evaluation('4c', args.overlapRatio, 2, args.precision, args.recall, args.threshold)