"""
Hint: please ingore the chinease annotations whcih may be wrong and they are just remains from old version.
"""

import argparse
import math
import os
import time
import warnings
from itertools import product
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from config.config import GetConfig, TrainingOpt
# from demo_image_orig import show_color_vector
from models.posenet import NetworkEval
from apex import amp

from utils.common import BodyPart, Human, CocoPart, CocoColors, CocoPairsRender
from utils.config_reader import config_reader
from utils.pafprocess import pafprocess
from utils.parse_skeletons import predict, find_peaks, find_connections, find_humans, heatmap_nms, predict_refactor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # choose the available GPUs
warnings.filterwarnings("ignore")

# For visualize
colors = [[128, 114, 250], [130, 238, 238], [48, 167, 238], [180, 105, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0],
          [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
          [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170],
          [255, 0, 85], [193, 193, 255], [106, 106, 255], [20, 147, 255]]

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='PoseNet Training')
parser.add_argument('--resume', '-r', action='store_true', default=True, help='resume from checkpoint')
parser.add_argument('--max_grad_norm', default=5, type=float,
                    help="If the norm of the gradient vector exceeds this, re-normalize it to have the norm equal to max_grad_norm")
parser.add_argument('--image', type=str, default='try_image/3_p.jpg', help='input image')  # required=True
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
flip_heat_ord = config.flip_heat_ord
flip_paf_ord = config.flip_paf_ord
draw_list = config.draw_list

NUM_KEYPOINTS = 18
RUN_REFACTOR = True
RUN_WITH_CPP = True


# ###############################################################################################################
def process(input_image_path, model, test_cfg, model_cfg, heat_layers, paf_layers):
    ori_img = cv2.imread(input_image_path)
    image_h, image_w, _ = ori_img.shape

    # > python
    if RUN_REFACTOR:
        # > [1]
        tic = time.time()
        heatmaps, pafs = predict_refactor(ori_img, model, test_cfg, model_cfg, input_image_path, flip_avg=True, config=config)
        all_peaks = heatmap_nms(heatmaps, model_cfg['stride'])
        toc = time.time()
        print('> [original drawing] heatmap elapsed = ', toc - tic)
        # `paf_upsamp`: (H, W, 30)
        pafs = cv2.resize(pafs, None,
                          fx=model_cfg['stride'],
                          fy=model_cfg['stride'],
                          interpolation=cv2.INTER_CUBIC)
    else:
        tic = time.time()
        # > [2] `heatmaps`, `pafs` are already upsampled in `predict()`
        heatmaps, pafs = predict(ori_img, model, test_cfg, model_cfg, input_image_path, flip_avg=True, config=config)
        all_peaks = find_peaks(heatmaps, test_cfg)
        toc = time.time()
        print('> [original drawing] heatmap elapsed = ', toc - tic)

    # > python
    if not RUN_WITH_CPP:
        # MY-TODO: refactor `show_color_vector` method.
        # show_color_vector(ori_img, pafs, heatmaps)

        connected_limbs, special_limb = find_connections(all_peaks, pafs, image_h, test_cfg, joint2limb_pairs)
        person_to_joint_assoc, joint_candidates = find_humans(connected_limbs, special_limb, all_peaks, test_cfg, joint2limb_pairs)

    canvas = cv2.imread(input_image)  # B,G,R order

    humans = []  # > (#person, [(x,y)*17, score])
    if RUN_REFACTOR:
        tic = time.time()
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
                        peak_id = int(pafprocess.get_part_peak_id(human_id, part_idx))
                        if peak_id < 0:
                            continue
                        is_added = True
                        human.body_parts[part_idx] = BodyPart(
                            '%d-%d' % (human_id, part_idx), part_idx,
                            # TOCHECK: divided by [H, W] ?
                            pafprocess.get_part_x(peak_id),
                            pafprocess.get_part_y(peak_id),
                            pafprocess.get_part_score(peak_id)
                        )
                    if is_added:
                        score = pafprocess.get_score(human_id)
                        human.score = score
                        humans.append(human)
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

        centers = {}
        for human in humans:
            # > draw points
            for i in range(CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue

                body_part = human.body_parts[i]
                # center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                center = (int(body_part.x), int(body_part.y))
                centers[i] = center
                cv2.circle(canvas, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

            # draw lines
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue

                cv2.line(canvas, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

        toc = time.time()
        print('> [original drawing] elapsed = ', toc - tic)
    else:
        tic = time.time()
        # `person_to_joint_assoc`: (#person, 20, 2)
        for person in person_to_joint_assoc[..., 0]:  # > (#person, 20)
            peak_ids = person[:NUM_KEYPOINTS]  # > (18,)
            person_keypoint_coordinates = []  # > (18, [x,y])
            for peak_id in peak_ids:
                if peak_id == -1:
                    # "No candidate for keypoint" # 標誌為-1的part是沒有檢測到的
                    X, Y = 0, 0
                else:
                    X, Y = joint_candidates[peak_id.astype(int), :2]  # > (#peaks, [x,y,score,id]) -> [x,y]
                person_keypoint_coordinates.append((X, Y))
            person_keypoint_coordinates_coco = [None] * 17  # (17,)
            # > TOCHECK: why use custom pairs instead of using coco pairs?
            for dt_index, gt_index in dt_gt_mapping.items():  # > (18,): {cmu_id:coco_id}
                if gt_index is None:
                    continue
                person_keypoint_coordinates_coco[gt_index] = person_keypoint_coordinates[dt_index]

            # person[-2] is the score
            humans.append((person_keypoint_coordinates_coco, 1 - 1.0 / person[-2]))

        # 畫所有的骨架
        color_board = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        color_idx = 0
        for i in draw_list:  # > #limbs
            for person_id, person in enumerate(person_to_joint_assoc):  # > (#person, 20, 2) -> #person
                peak_ids = person[joint2limb_pairs[i], 0].astype(int)  # (2, 2) -> (2,)
                if -1 in peak_ids:  # 有-1說明沒有對應的關節點與之相連,即有一個類型的part沒有缺失，無法連接成limb
                    continue
                # 在上一個cell中有　canvas = cv2.imread(test_image) # B,G,R order
                cur_canvas = canvas.copy()
                Xs = joint_candidates[peak_ids, 0]  # > (#peaks, [x,y,score,id]) -> (x1,x2)
                Ys = joint_candidates[peak_ids, 1]  # > (#peaks, [x,y,score,id]) -> (y1,y2)
                mXs = np.mean(Xs)
                mYs = np.mean(Ys)
                length = ((Ys[0] - Ys[1]) ** 2 + (Xs[0] - Xs[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Ys[0] - Ys[1], Xs[0] - Xs[1]))
                polygon = cv2.ellipse2Poly((int(mXs), int(mYs)), (int(length / 2), 3), int(angle), 0, 360, 1)

                cv2.circle(cur_canvas, (int(Xs[0]), int(Ys[0])), 4, color=[0, 0, 0], thickness=2)
                cv2.circle(cur_canvas, (int(Xs[1]), int(Ys[1])), 4, color=[0, 0, 0], thickness=2)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[color_board[color_idx]])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            color_idx += 1
        toc = time.time()
        print('> [original drawing] elapsed = ', toc - tic)
    return canvas


def show_color_vector(oriImg, paf_avg, heatmap_avg):
    hsv = np.zeros_like(oriImg)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(paf_avg[:, :, 16], 1.5 * paf_avg[:, :, 16])  # 設置不同的係數，可以使得顯示顏色不同

    # 將弧度轉換為角度，同時OpenCV中的H範圍是180(0 - 179)，所以再除以2
    # 完成後將結果賦給HSV的H通道，不同的角度(方向)以不同顏色表示
    # 對於不同方向，產生不同色調
    # hsv[...,0]等價於hsv[:,:,0]
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 將矢量大小標準化到0-255範圍。因為OpenCV中V份量對應的取值範圍是256
    # 對於同一H、S而言，向量的大小越大，對應顏色越亮
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # 最後，將生成好的HSV圖像轉換為BGR顏色空間
    limb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(limb_flow, alpha=.5)
    plt.show()

    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(paf_avg[:, :, 11], alpha=.6)
    plt.show()

    plt.imshow(heatmap_avg[:, :, -1])
    plt.imshow(oriImg[:, :, [2, 1, 0]], alpha=0.25)  # show a keypoint
    plt.show()

    plt.imshow(heatmap_avg[:, :, -2])
    plt.imshow(oriImg[:, :, [2, 1, 0]], alpha=0.5)  # show the person mask
    plt.show()

    plt.imshow(oriImg[:, :, [2, 1, 0]])  # show a keypoint
    plt.imshow(heatmap_avg[:, :, 4], alpha=.5)
    plt.show()
    t = 2


if __name__ == '__main__':
    input_image = args.image
    output = args.output

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
    tic = time.time()
    with torch.no_grad():
        canvas = process(input_image,
                         posenet,
                         params,
                         model_params,
                         config.heat_layers + 2,
                         config.paf_layers)  # todo background + 2

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.show()
