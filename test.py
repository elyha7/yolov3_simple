from detector import YoloV3Predictor
import sys
from PIL import Image
import numpy as np
import json
import random
from utils.utils import plot_one_box


if __name__ == '__main__':
    model = YoloV3Predictor(0, 416, half_precision=False)
    if len(sys.argv) == 1:
        img = np.array(Image.open('data/samples/bus.jpg'))
    else:
        img = np.array(Image.open(sys.argv[1]))

    res = model.predict(img,conf_thres=0.5,agnostic_iou=False)

    with open('cfg/labels.json', 'r') as f:
        label_dict = json.load(f)
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(label_dict))]
    for *xyxy, conf, cls in res[0]:
        label = '%s %.2f' % (list(label_dict.keys())[int(cls)], conf)
        plot_one_box(xyxy, img, label=label, color=colors[int(cls)])
    Image.fromarray(img).save('output/result.jpg')
