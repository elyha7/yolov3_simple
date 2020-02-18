# YoloV3 simple

## Description

This project is wrap over [ultralytics](https://github.com/ultralytics/yolov3) open source implementation of yolo v3. The idea is to provide developers with easy interface to apply detector in different sorts of python computer vision applications.

## Installation

* Install requirements: `pip3 install -r requirements.txt`
* Download model weights (about 250mb) from [google drive](https://drive.google.com/drive/folders/1B9FWmb6JkGV44C3EP8fuDTlX9gqnRzLK?usp=sharing)
* Put weights in `weights/` folder

## Usage

```python

from PIL import Image
from detector import YoloV3Predictor
import numpy as np

model = YoloV3Predictor('cpu', 416)
img = np.array(Image.open('data/samples/bus.jpg'))
res = model.predict(img,conf_thres=0.5,agnostic_iou=False)
```
Command line example:
`python test.py data/samples/bus.jpg`
the result would appear in `output/` folder

![Output example](/output/result.jpg)
## Citiation
Thanks [ultralytics](https://github.com/ultralytics/yolov3) for pretrained models.
[![DOI](https://zenodo.org/badge/146165888.svg)](https://zenodo.org/badge/latestdoi/146165888)
