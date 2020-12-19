import os
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
import numpy as np
import torch
import mmcv
import cv2
import json
import PIL


testset_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2019ArT/test_task13'
output_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2019ArT/output/preds'

model_name = 'solo_r50_fpn_1x_coco'
config_file = 'work_dirs/'+model_name+'/'+model_name+'.py'
checkpoint_file = 'work_dirs/'+model_name+'/latest.pth'
print(f'inferencing using model: {checkpoint_file}')
model = init_detector(config_file, checkpoint_file, device='cuda:0')
score_thr = 0.3

print('start inference')
for image in tqdm(os.listdir(testset_dir)):
    image_path = os.path.join(testset_dir, image)

    try:
        im = PIL.Image.open(image_path)
        im.close()
    except PIL.Image.DecompressionBombError:
        print(f'skip: {image_path}')
        continue

    image_index = image.split('.')[0].split('_')[1]

    result = inference_detector(model, image_path)
    torch.cuda.empty_cache()

    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    preds = []
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        for i in inds:
            i = int(i)
            sg = segms[i]
            if isinstance(sg, torch.Tensor):
                sg = sg.detach().cpu().numpy()
            mask = sg.astype(np.uint8)
            mask *= 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = [[float(point[0][0]), float(point[0][1])]
                      for point in contours[0]]
            confidence = bboxes[i][-1]
            preds.append({
                'points': points,
                'confidence': float(confidence)
            })

    output_file = os.path.join(output_dir, image_index+'.json')
    with open(output_file, 'w')as f:
        json.dump(preds, f)

print('collecting results')
submit = dict()
submit_file = '/home/xiekaiyu/ocr/dataset/ICDAR2019ArT/output/submit.json'
for pred in tqdm(os.listdir(output_dir)):
    pred_path = os.path.join(output_dir, pred)
    image_index = pred.split('.')[0]
    with open(pred_path, 'r')as f:
        result = json.load(f)
        submit['res_'+image_index] = result

# skip image
submit['res_3102'] = [{
    'points': [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    'confidence':0.0
}]

with open(submit_file, 'w')as f:
    json.dump(submit, f)
