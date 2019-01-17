# mAp_cal
mAp calculation tools
## Introduction
Lots of object detection algorithms used mAp(mean Average precision) as evaluation metric, and yet their mAp calculation 
code embedded in algorithm implementation code, which is not easy to be separated. Here is offering an independent mAp 
calculation tool, you can do:
- Calculating mAp for your object detection algorithm testing
- Get recall and precision in any confidence threshold 
- Drawing PR curve
- Drawing ROC curve(coming soon)  
- Get FPPI(False Positive Per Image) and FPPW(False Positive Per Window) in any threshold(coming soon) 
## Requirements
- python 2.X OR python 3.X
- python-opencv (any version)
- matplotlib 2.2.3
## Input format
coming soon :-)
## Usage
```shell
python test.py --dir ./gt ./predict --ratio 0.5 --thre 0.7 --cls 2 --prec 0 --rec 0
```

