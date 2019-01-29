# Object Detection evaluation tools
Object Detection evaluation tools
## Introduction
Lots of object detection algorithms used mAp(mean Average precision) as evaluation metric, and yet their mAp calculation 
code embedded in algorithm implementation code, which is not easy to be separated. Here is offering an independent mAp 
calculation tool, you can do:
- Calculating mAp for your object detection algorithm testing(11 points method supported)
- Get recall and precision in any confidence threshold 
- Drawing PR curve
- Drawing ROC curve(Done)  
- Get FPPI(False Positive Per Image) and FPPW(False Positive Per Window) in any threshold(Done) 
## Requirements
- python 2.X OR python 3.X
- python-opencv (any version)
- matplotlib 2.2.3
## Input format
- TXT format, see 'sample' folder
- XML format, only Pascal VOC style supported
## Usage
```shell
python test.py --dir ./sample/prediction ./sample/test_annos --ratio 0.5 --thre 0.7
```
Anything you want to know about usage can be found with typing:
```shell
python test.py --h
```
or
```shell
python test.py --help
```
