from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import numpy as np
import os
from tqdm import tqdm
import torch
import PIL
import cv2


# Specify the path to model config and checkpoint file
model_name = 'solo_r50_fpn_1x_coco'
if 'polar' in model_name:
    from mmdet.apis import polar_inference_detector as inference_detector

config_file = 'work_dirs/'+model_name+'/'+model_name+'.py'
checkpoint_file = 'work_dirs/'+model_name+'/latest.pth'

print(f'inferencing using model: {checkpoint_file}')

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

image_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2015TextLocalization/test_img'
output_image_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2015TextLocalization/script_test_ch4_t1_e1-1577983151/output'
res_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2015TextLocalization/script_test_ch4_t1_e1-1577983151/submit'
score_thr = 0.3


def eval_icdar2015():
    print('eval icdar2013')
    images = os.listdir(image_dir)
    print(f'images: {len(images)}')

    for image in tqdm(images):
        image_name = image.split('.')[0]
        res = os.path.join(res_dir, 'res_'+image_name+'.txt')
        image = os.path.join(image_dir, image)

        result = inference_detector(model, image)
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
        img = cv2.imread(image)
        with open(res, 'w') as f:
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

                    # horizontal box (green)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # angled box (red)
                    rect = cv2.minAreaRect(contours[0])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

                    points = []
                    for p in box:
                        points.append(str(p[0]))
                        points.append(str(p[1]))

                    line = ','.join(points)
                    f.write(line+'\n')

        cv2.imwrite(os.path.join(output_image_dir, 'res_'+image_name+'.jpg'),
                    img)

    print('eval done')


if __name__ == "__main__":
    eval_icdar2015()
