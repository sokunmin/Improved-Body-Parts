"""
Hint: please ingore the chinease annotations whcih may be wrong and they are just remains from old version.
"""

import sys

from utils.common import Human, BodyPart
from utils.pafprocess import pafprocess

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

ORDER_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
# ORDER_COCO = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
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
NUM_COCO_KEYPOINTS = 17
RUN_REFACTOR = False
RUN_WITH_CPP = False
TEST_SET = 'val2017'


# ######################################  For evaluating time ######################################
def process(input_image_path, model, test_cfg, model_cfg, heat_layers, paf_layers):

    ori_img = cv2.imread(input_image_path)
    img_h, img_w, _ = ori_img.shape
    if RUN_REFACTOR:
        heatmaps, pafs = predict_refactor(ori_img, model, test_cfg, model_cfg, input_image_path, flip_avg=True, config=config)
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
    if not RUN_WITH_CPP:
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
            if RUN_WITH_CPP:
                # `heatmap_upsamp`: (H, W, 19)
                heatmap_upsamp = cv2.resize(
                    heatmaps, None,
                    fx=model_cfg['stride'],
                    fy=model_cfg['stride'],
                    interpolation=cv2.INTER_CUBIC)
                # `joint_list`: (#person * 18, 5)
                joint_list = np.expand_dims(joint_list, 0)
                paf_upsamp = pafs
                pafprocess.process_paf(joint_list, heatmap_upsamp, paf_upsamp)
                for human_id in range(pafprocess.get_num_humans()):
                    human = Human([])
                    is_added = False
                    for part_idx in range(NUM_KEYPOINTS):
                        c_idx = int(pafprocess.get_part_peak_id(human_id, part_idx))
                        if c_idx < 0:
                            continue
                        is_added = True
                        human.body_parts[part_idx] = BodyPart(
                            '%d-%d' % (human_id, part_idx), part_idx,
                            pafprocess.get_part_x(c_idx),
                            pafprocess.get_part_y(c_idx),
                            pafprocess.get_part_score(c_idx)
                        )
                    if is_added:
                        score = pafprocess.get_score(human_id)
                        human.score = score
                        humans.append(human)
            else:
                # > python
                for person_id, person in enumerate(person_to_joint_assoc[..., 0]):
                    peak_ids = person[:NUM_KEYPOINTS]  # > (18,)
                    human = Human([])
                    is_added = False
                    for part_idx, peak_id in enumerate(peak_ids):  # > #kp
                        if peak_id < 0:
                            continue
                        is_added = True
                        x, y, peak_score = joint_candidates[peak_id.astype(int), :3]
                        human.body_parts[part_idx] = BodyPart(
                            '%d-%d' % (person_id, part_idx),
                            part_idx,
                            x, y,
                            peak_score
                        )
                        # print('> [', person_id, '] / [', peak_id, '] = ', (x, y))
                    if is_added:
                        limb_score = person[-2]
                        # TOCHECK: 1 - 1.0 / person[-2]
                        # limb_score = 1 - 1.0 / person[-2]
                        human.score = limb_score
                        humans.append(human)

    else:
        # `person_to_joint_assoc`: (#person, 20, 2)
        for person_id, person in enumerate(person_to_joint_assoc[..., 0]):  # > (#person, 20)
            peak_ids = person[:NUM_KEYPOINTS]  # > (18,)
            person_keypoint_coordinates = []
            for peak_id in peak_ids:
                if peak_id == -1:
                    # "No candidate for keypoint" # 標誌為-1的part是沒有檢測到的
                    x, y, v = 0, 0, 0
                else:
                    x, y = joint_candidates[peak_id.astype(int), :2]
                    v = 1 if x > 0 or y > 0 else 0
                person_keypoint_coordinates.append((x, y, v))

            coco_keypoints = np.array(person_keypoint_coordinates)[ORDER_COCO, :]

            # person[-2] is the score,
            humans.append((coco_keypoints, 1 - 1.0 / person[-2]))  # TOCHECK: 1-(1/x)?
    return humans


def get_image_name(coco, image_id):
    return coco.imgs[image_id]['file_name']


def append_result(image_id, humans, all_outputs):
    if RUN_REFACTOR:
        for human in humans:
            one_result = {
                "image_id": 0,
                "category_id": 1,
                "keypoints": [],
                "score": 0
            }
            cmu_keypoints = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)

            for i in range(NUM_KEYPOINTS):
                if i not in human.body_parts.keys():
                    cmu_keypoints[i, 0] = 0
                    cmu_keypoints[i, 1] = 0
                    cmu_keypoints[i, 2] = 0
                else:
                    body_part = human.body_parts[i]
                    cmu_keypoints[i, 0] = body_part.x
                    cmu_keypoints[i, 1] = body_part.y
                    cmu_keypoints[i, 2] = 1

            one_result["image_id"] = image_id
            one_result["score"] = 1.
            cmu_keypoints = cmu_keypoints[ORDER_COCO, :]
            one_result["keypoints"] = list(cmu_keypoints.reshape(NUM_KEYPOINTS * 3))

            all_outputs.append(one_result)
    else:
        # Question: do we need to sort the detections by scores before evaluation ?
        # -- I think we do not have. COCO will select the top 20 automatically

        for keypoint_list, score in humans:
            one_result = {
                "image_id": 0,
                "category_id": 1,
                "keypoints": [],
                "score": 0
            }
            coco_keypoints = np.zeros((NUM_COCO_KEYPOINTS, 3), dtype=np.float64)

            for i, xyv in enumerate(keypoint_list):
                coco_keypoints[i, 0] = xyv[0]
                coco_keypoints[i, 1] = xyv[1]
                coco_keypoints[i, 2] = xyv[2]

            one_result['image_id'] = image_id
            one_result['keypoints'] = list(coco_keypoints.reshape(NUM_COCO_KEYPOINTS * 3))
            one_result['score'] = score  # NOTE: `score` must be assigned accordingly.

            all_outputs.append(one_result)


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

    results_file = 'results/%s_%s_results.json' % (valset_name, dump_name)
    print('the path of detected keypoint file is: ', results_file)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    assert (not set(img_ids).difference(set(cocoGt.getImgIds())))
    tic = time.time()
    all_outputs = []
    for image_id in tqdm.tqdm(img_ids):
        image_name = get_image_name(cocoGt, image_id)
        image_name = os.path.join(dataDir, img_subdir, image_name)
        humans = process(image_name, model, dict(params), dict(model_params), config.heat_layers + 2, config.paf_layers)

        append_result(image_id, humans, all_outputs)

    with open(results_file, 'w') as f:
        json.dump(all_outputs, f)

    toc = time.time()
    print('> json processing time is %.5f' % (toc - tic))
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
    # print('> Model = ', posenet)
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
