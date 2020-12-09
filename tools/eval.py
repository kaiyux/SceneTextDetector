from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import os
from tqdm import tqdm


image_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2015TextLocalization/test_img'
output_image_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2015TextLocalization/script_test_ch4_t1_e1-1577983151/output'
res_dir = '/home/xiekaiyu/ocr/dataset/ICDAR2015TextLocalization/script_test_ch4_t1_e1-1577983151/submit'

config_file = 'configs/ocr/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py'
checkpoint_file = 'work_dirs/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco/epoch_4.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')


def eval_icdar2015():
    print('eval icdar2015')
    images = os.listdir(image_dir)
    print(f'images: {len(images)}')

    for image in tqdm(images):
        image_name = image.split('.')[0]
        res = os.path.join(res_dir, 'res_'+image_name+'.txt')
        image = os.path.join(image_dir, image)

        result = inference_detector(model, image)
        model.show_result(image, result,
                          out_file=os.path.join(output_image_dir, 'res_'+image_name+'.jpg'))
        bboxes = np.vstack(result)  # upper left & right bottom
        with open(res, 'w') as f:
            for bbox in bboxes:
                bbox_int = bbox.astype(np.int32)
                left_top = [bbox_int[0], bbox_int[1]]
                right_bottom = [bbox_int[2], bbox_int[3]]
                right_top = [bbox_int[2], bbox_int[1]]
                left_bottom = [bbox_int[0], bbox_int[3]]
                corners = [left_top, right_top, right_bottom, left_bottom]
                points = []
                for p in corners:
                    points.append(str(p[0]))
                    points.append(str(p[1]))

                line = ','.join(points)
                f.write(line+'\n')

    print('eval done')


def eval_icdar2013():
    print('eval icdar2013')
    images = os.listdir(image_dir)
    print(f'images: {len(images)}')

    for image in tqdm(images):
        image_name = image.split('.')[0]
        res = os.path.join(res_dir, 'res_'+image_name+'.txt')
        image = os.path.join(image_dir, image)

        result = inference_detector(model, image)
        model.show_result(image, result,
                          out_file=os.path.join(output_image_dir, 'res_'+image_name+'.jpg'))
        bboxes = np.vstack(result)  # upper left & right bottom
        with open(res, 'w') as f:
            for bbox in bboxes:
                bbox_int = bbox.astype(np.int32)
                left_top = [bbox_int[0], bbox_int[1]]
                right_bottom = [bbox_int[2], bbox_int[3]]
                corners = [left_top, right_bottom]
                points = []
                for p in corners:
                    points.append(str(p[0]))
                    points.append(str(p[1]))

                line = ','.join(points)
                f.write(line+'\n')

    print('eval done')


if __name__ == "__main__":
    # eval_icdar2013()
    eval_icdar2015()
