"""
Hint: please ingore the chinease annotations whcih may be wrong and they are just remains from old version.
"""

import sys

from utils.common import Human, BodyPart

sys.path.append("..")  # 包含上级目录
import argparse
import math
import os
import time
import warnings
from itertools import product
import numpy as np
import cv2
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.config_reader import config_reader
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
from models.posenet import NetworkEval
from apex import amp
from utils.parse_skeletons import predict, find_peaks, find_connections, find_humans, predict_refactor, heatmap_nms
import tqdm
import json
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # choose the available GPUs
warnings.filterwarnings("ignore")

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [193, 193, 255], [106, 106, 255], [20, 147, 255],
          [128, 114, 250], [130, 238, 238], [48, 167, 238], [180, 105, 255]]

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
parser.add_argument('--checkpoint_path', '-p', default='checkpoints_parallel', help='save path')
parser.add_argument('--max_grad_norm', default=5, type=float,
                    help="If the norm of the gradient vector exceeds this, re-normalize it to have the norm equal to max_grad_norm")
parser.add_argument('--output', type=str, default='result.jpg', help='output image')

parser.add_argument('--opt-level', type=str, default='O1')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

args = parser.parse_args()

# ###################################  Setup for some configurations ###########################################
opt = TrainingOpt()
config = GetConfig(opt.config_name)

joint2limb_pairs = config.limbs_conn  # > 30
dt_gt_mapping = config.dt_gt_mapping
NUM_KEYPOINTS = 18
RUN_REFACTOR = True
RUN_WITH_CPP = False
TEST_SET = 'val2017'


# ######################################  For evaluating time ######################################
def process(input_image_path, model, test_cfg, model_cfg, heat_layers, paf_layers):

    ori_img = cv2.imread(input_image_path)
    img_h, img_w, _ = ori_img.shape
    if RUN_REFACTOR:
        heatmaps, pafs = predict_refactor(ori_img, model, test_cfg, model_cfg, input_image_path, True, config)
        all_peaks = heatmap_nms(heatmaps, model_cfg['stride'])
        pafs = cv2.resize(pafs, None,
                          fx=model_cfg['stride'],
                          fy=model_cfg['stride'],
                          interpolation=cv2.INTER_CUBIC)
    else:
        # > [1] original
        heatmaps, pafs = predict(ori_img, model, test_cfg, model_cfg, input_image_path, flip_avg=True, config=config)
        all_peaks = find_peaks(heatmaps, test_cfg)

    end = time.time()

    connected_limbs, special_limb = find_connections(all_peaks, pafs, img_h, test_cfg, joint2limb_pairs)
    person_to_joint_assoc, joint_candidates = find_humans(connected_limbs, special_limb, all_peaks, test_cfg, joint2limb_pairs)

    batch_time.update((time.time() - end))
    if show_eval_speed:
        print('==================>Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Speed {2:.3f} ({3:.3f})\t'.format(1, 1, 1 / batch_time.val, 1 / batch_time.avg, batch_time=batch_time))

    humans = []
    if RUN_REFACTOR:  # Refactored
        joint_list = np.array([
            tuple(peak) + (joint_type,)
            for joint_type, joint_peaks in enumerate(all_peaks)
            for peak in joint_peaks
        ]).astype(np.float32)

        if joint_list.shape[0] > 0:
            # `heatmap_upsamp`: (H, W, 19)
            heatmap_upsamp = cv2.resize(
                heatmaps, None,
                fx=config.MODEL.DOWNSAMPLE,
                fy=config.MODEL.DOWNSAMPLE,
                interpolation=cv2.INTER_NEAREST)
            if RUN_WITH_CPP:
                # `joint_list`: (#person * 18, 5)
                joint_list = np.expand_dims(joint_list, 0)
                # `paf_upsamp`: (H, W, 38)
                paf_upsamp = cv2.resize(
                    pafs, None,
                    fx=config.MODEL.DOWNSAMPLE,
                    fy=config.MODEL.DOWNSAMPLE,
                    interpolation=cv2.INTER_NEAREST)
            else:
                # > python
                for person_id, person in enumerate(person_to_joint_assoc[..., 0]):
                    human = Human([])
                    is_added = False
                    peak_ids = person[:NUM_KEYPOINTS]  # > (18,)
                    for part_idx, peak_id in enumerate(peak_ids):  # > #kp
                        if peak_id < 0:
                            continue
                        is_added = True
                        x, y, peak_score = joint_candidates[peak_id.astype(int), :3]
                        x = float(x / heatmap_upsamp.shape[1])
                        y = float(y / heatmap_upsamp.shape[0])
                        human.body_parts[part_idx] = BodyPart(
                            '%d-%d' % (person_id, part_idx),
                            part_idx,
                            x, y,
                            peak_score
                        )
                    if is_added:
                        limb_score = person[-2]
                        # TOCHECK: 1 - 1.0 / person[-2]
                        # limb_score = 1 - 1.0 / person[-2]
                        human.score = limb_score
                        humans.append(human)
    else:
        # `person_to_joint_assoc`: (#person, 20, 2)
        for person in person_to_joint_assoc[..., 0]:  # > (#person, 20)
            peak_ids = person[:NUM_KEYPOINTS]  # > (18,)
            person_keypoint_coordinates = []
            for peak_id in peak_ids:
                if peak_id == -1:
                    # "No candidate for keypoint" # 標誌為-1的part是沒有檢測到的
                    X, Y = 0, 0
                else:
                    X, Y = joint_candidates[peak_id.astype(int), :2]
                person_keypoint_coordinates.append((X, Y))
            person_keypoint_coordinates_coco = [None] * 17
            # > TOCHECK: why use custom pairs instead of using coco pairs?
            for dt_index, gt_index in dt_gt_mapping.items():
                if gt_index is None:
                    continue
                person_keypoint_coordinates_coco[gt_index] = person_keypoint_coordinates[dt_index]

            # TOCHECK: 1-(1/x)?,
            # person[-2] is the score
            humans.append((person_keypoint_coordinates_coco, 1 - 1.0 / person[-2]))
    return humans


def get_image_name(coco, image_id):
    return coco.imgs[image_id]['file_name']


def predict_many(coco, images_directory, validation_ids, params, model, model_params, heat_layers, paf_layers):
    assert (not set(validation_ids).difference(set(coco.getImgIds())))

    keypoints = {}

    for image_id in tqdm.tqdm(validation_ids):
        image_name = get_image_name(coco, image_id)
        image_name = os.path.join(images_directory, image_name)
        keypoints[image_id] = process(image_name, model, dict(params), dict(model_params), heat_layers + 2, paf_layers)
        # fixme: heat_layers + 1 if you use background keypoint  !!!
    return keypoints


def format_results(humans, resFile):
    if RUN_REFACTOR:
        pass
    else:
        format_keypoints = []
        # Question: do we need to sort the detections by scores before evaluation ?
        # -- I think we do not have. COCO will select the top 20 automatically
        for image_id, people in humans.items():  # > {image_id: (#person, [(x,y)*17, score])}
            for keypoint_list, score in people:
                format_keypoint_list = []  # [x,y,v] * #kp
                for x, y in keypoint_list:
                    for v in [x, y, 1 if x > 0 or y > 0 else 0]:  #
                        # 坐标取了整数,为了减少文件的大小，如果x,y有一个有值，那么标记这个点为可见。　如果x or y =0,令v=0,coco只评测v>0的点
                        format_keypoint_list.append(v)

                format_keypoints.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": format_keypoint_list,
                    "score": score,
                })

        json.dump(format_keypoints, open(resFile, 'w'))


def validation(model, dump_name, img_subdir):
    # `img_subdir`: [val2017, test2017]
    dataDir = 'data/dataset/coco'
    annType = 'keypoints'
    if 'val' in img_subdir:
        # For evaluation on validation set
        valset_name = 'person_keypoints_val2017'
    else:
        # For evaluation on test-dev set
        valset_name = 'image_info_test-dev2017'
    ann_file = '%s/annotations/%s.json' % (dataDir, valset_name)
    print('> ann_file=', ann_file)

    cocoGt = COCO(ann_file)
    # cat_ids = cocoGt.getCatIds(catNms=['person'])
    # img_ids = cocoGt.getImgIds(catIds=cat_ids)
    img_ids = cocoGt.getImgIds()[:500]

    results_file = '%s/results/%s_%s_results.json'
    results_file = results_file % (dataDir, valset_name, dump_name)
    print('the path of detected keypoint file is: ', results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    assert (not set(img_ids).difference(set(cocoGt.getImgIds())))

    keypoints = {}

    for image_id in tqdm.tqdm(img_ids):
        image_name = get_image_name(cocoGt, image_id)
        image_name = os.path.join(dataDir, img_subdir, image_name)
        keypoints[image_id] = process(
            image_name, model, dict(params), dict(model_params), config.heat_layers + 2, config.paf_layers)

    format_results(keypoints, results_file)
    cocoDt = cocoGt.loadRes(results_file)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval


# ###############################################################################################################


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


batch_time = AverageMeter()

if __name__ == "__main__":
    posenet = NetworkEval(opt, config, bn=True)
    print('> Model = ', posenet)
    print('Resuming from checkpoint ...... ')
    checkpoint = torch.load(opt.ckpt_path, map_location=torch.device('cpu'))  # map to cpu to save the gpu memory
    posenet.load_state_dict(checkpoint['weights'])  # 加入他人訓練的模型，可能需要忽略部分層，則strict=False
    print('Network weights have been resumed from checkpoint...')

    if torch.cuda.is_available():
        posenet.cuda()

    posenet = amp.initialize(posenet,
                             opt_level=args.opt_level,
                             keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                             loss_scale=args.loss_scale)
    posenet.eval()  # set eval mode is important

    # load config
    params, model_params = config_reader()

    # show keypoint assignment algorithm speed
    show_eval_speed = False
    tic = time.time()
    with torch.no_grad():
        eval_result_original = validation(posenet,
                                          dump_name='residual_4_hourglass_focal_epoch_52_512_input_1scale_max',
                                          img_subdir=TEST_SET)  # ['val2017', 'test2017']
    toc = time.time()
    print('processing time is %.5f' % (toc - tic))
    #  若是在test数据集上进行预测并写结果，则
    # annFile='keras_Realtime_Multi-Person_Pose_Estimation-new-generation/dataset/coco/link2coco2017/annotations_trainval_info/image_info_test2017.json'
    # cocoGt = COCO(annFile)
    # validation_ids = cocoGt.getImgIds() 将获得带有image id的一个list
