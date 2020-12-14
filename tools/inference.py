from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import os
from tqdm import tqdm


# Specify the path to model config and checkpoint file
config_file = 'work_dirs/yolact_r101_1x8_coco/yolact_r101_1x8_coco.py'
checkpoint_file = 'work_dirs/yolact_r101_1x8_coco/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


image_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2015TextLocalization/test_img'
output_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2015TextLocalization/output/yolact_r101_1x8_coco'

for img in tqdm(os.listdir(image_dir)):
    image_path = os.path.join(image_dir, img)
    output_path = os.path.join(output_dir, img)
    result = inference_detector(model, image_path)
    model.show_result(image_path, result, out_file=output_path)
