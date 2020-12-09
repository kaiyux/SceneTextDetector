from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np


# Specify the path to model config and checkpoint file
config_file = 'configs/ocr/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py'
checkpoint_file = 'work_dirs/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco/epoch_4.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# or img = mmcv.imread(img), which will only load it once
img = '/home/xiekaiyu/ocr/dataset/ICDAR2015TextLocalization/test_img/img_10.jpg'
result = inference_detector(model, img)
bboxes = np.vstack(result)
print(bboxes)
# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')
