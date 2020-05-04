"""
Hint: please ingore the chinease annotations whcih may be wrong and they are just remains from old version.
"""

import sys

sys.path.append("..")  # 包含上级目录
import json
import math
import numpy as np
from itertools import product
import tqdm
import time
import cv2
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.config_reader import config_reader
from utils import util
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
import matplotlib.pyplot as plt
from models.posenet import NetworkEval
import warnings
import os
import argparse
from apex import amp


os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # choose the available GPUs
warnings.filterwarnings("ignore")

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [193, 193, 255], [106, 106, 255], [20, 147, 255],
          [128, 114, 250], [130, 238, 238], [48, 167, 238], [180, 105, 255]]

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
flip_heat_ord = config.flip_heat_ord
flip_paf_ord = config.flip_paf_ord


# ######################################  For evaluating time ######################################
def process(input_image_path, params, model, model_params, heat_layers, paf_layers):
    oriImg = cv2.imread(input_image_path)  # B,G,R order.    训练数据的读入也是用opencv，因此也是B, G, R顺序
    img_h, img_w, _ = oriImg.shape
    torch.cuda.empty_cache()
    heatmap, paf = predict(oriImg, params, model_params, heat_layers, paf_layers, input_image_path)
    end = time.time()  # ############# Evaluating the keypoint assignment algorithm ######

    all_peaks = find_peaks(heatmap, params)
    connection_all, special_k = find_connections(all_peaks, paf, img_h, params)
    subset, candidate = find_people(connection_all, special_k, all_peaks, params)

    batch_time.update((time.time() - end))
    if show_eval_speed:
        print('==================>Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Speed {2:.3f} ({3:.3f})\t'.format(1, 1, 1 / batch_time.val, 1 / batch_time.avg, batch_time=batch_time))

    keypoints = []
    # > `subset`: (#person, 20, 2)，MY-TODO 收集了屬於各個人物的keypoints
    for s in subset[..., 0]:   # > (#person, 20) -> #person
        keypoint_indexes = s[:18]  # 定义的keypoint一共有18个
        person_keypoint_coordinates = []  # > (#kp, (x,y))
        for index in keypoint_indexes:  # > (#kp,) -> kp_idx
            if index == -1:
                # "No candidate for keypoint" # 标志为-1的part是没有检测到的
                X, Y = 0, 0
            else:
                # > `candidate`: (#kp * #person, (x,y,score,id))
                X, Y = candidate[index.astype(int)][:2]
            person_keypoint_coordinates.append((X, Y))
        person_keypoint_coordinates_coco = [None] * 17
        # > TOCHECK: why use custom pairs instead of using coco pairs?
        for dt_index, gt_index in dt_gt_mapping.items():
            if gt_index is None:
                continue
            person_keypoint_coordinates_coco[gt_index] = person_keypoint_coordinates[dt_index] # > (x,y)

        # TOCHECK: 1-(1/x)?,
        # s[18] is the score, s[19] is the number of keypoint
        keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[18]))
    return keypoints


def predict(image, params, model_params, heat_layers, paf_layers, input_image_path, flip_avg=True):
    # > scale feature maps up to image size
    img_h, img_w, _ = image.shape
    heatmap_avg = np.zeros((img_h, img_w, heat_layers))  # > `heatmap_avg`: (imgH, imgW, 20)
    paf_avg = np.zeros((img_h, img_w, paf_layers))  # > `paf_layers`: (imgH, imgW, 30)
    # > [1] scale search
    multiplier = [x * model_params['boxsize'] / img_h for x in params['scale_search']]  # 把368boxsize去掉,效果稍微下降了
    # > [2] fix scale
    multiplier = [1.]  # > [0.5, 1., 1.5, 2., 3.]
    rotate_angle = params['rotation_search']  # > 0.0
    for item in product(multiplier, rotate_angle):
        scale, angle = item
        img_max_h, img_max_w = (2600, 3800)  # CHANGED: (2300, 3200)->(2600,3800)
        if scale * img_h > img_max_h or scale * img_w > img_max_w:
            scale = min(img_max_h / img_h, img_max_w / img_w)
            print("Input image: '{}' is too big, shrink it!".format(input_image_path))

        # > `imageToTest`: (scaleH, scaleW, 3)
        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # > `imageToTest_padded`: (scale_padH, scale_padW, 3)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest,
                                                          model_params['max_downsample'],  # > 64
                                                          model_params['padValue'])  # > 128
        scale_padH, scale_padW, _ = imageToTest_padded.shape

        # > WARN: `[1-1]`: we use OpenCV to read image`(BGR)` all the time
        # Input Tensor: a batch of images within [0,1], required shape in this project : (1, height, width, channels)
        input_img = np.float32(imageToTest_padded / 255)

        # > `[1-2]` :add rotate image
        if angle != 0:  # ADDED
            rotate_matrix = cv2.getRotationMatrix2D((input_img.shape[0] / 2, input_img.shape[1] / 2), angle, 1)
            rotate_matrix_reverse = cv2.getRotationMatrix2D((input_img.shape[0] / 2, input_img.shape[1] / 2), -angle, 1)
            input_img = cv2.warpAffine(input_img, rotate_matrix, (0, 0))

        # input_img -= np.array(config.img_mean[::-1])  # Notice: OpenCV uses BGR format, reverse the last axises
        # input_img /= np.array(config.img_std[::-1])

        # > `[1-2]` :add flip image
        swap_image = input_img[:, ::-1, :].copy()
        # plt.imshow(swap_image[:, :, [2, 1, 0]])  # Opencv image format: BGR
        # plt.show()
        input_img = np.concatenate((input_img[None, ...], swap_image[None, ...]), axis=0)  # (2, H, W, C)
        input_img = torch.from_numpy(input_img).cuda()

        # > `[1-3]-model`(4,)=(2, 50, featH, featW) x 4, `dtype=float16`
        output_tuple = posenet(input_img)  # > NOTE: feed img here -> (#stage, #scales, #img, 50, H, W)

        # > `[1-4]`: scales vary according to input image size.
        # > `-1`: last stage, `0`: high-res featmaps
        output = output_tuple[-1][0].cpu().numpy()  # -`> (2, 50, featH, featW)

        output_blob = output[0].transpose((1, 2, 0))  # > (featH, featW, 50)
        output_paf = output_blob[:, :, :config.paf_layers]  # > `PAF`:(featH, featW, 30)
        output_kp = output_blob[:, :, config.paf_layers:config.num_layers]  # > `KP`:(featH, featW, 20)
        # > flipped image output
        output_blob_flip = output[1].transpose((1, 2, 0))  # > (featH, featW, 50)
        output_paf_flip = output_blob_flip[:, :, :config.paf_layers]  # `PAF`: (featH, featW, 30)
        output_kp_flip = output_blob_flip[:, :, config.paf_layers:config.num_layers]  # > `KP`: (featH, featW, 20)

        # > `[1-5]`: flip ensemble & average
        if flip_avg:
            output_paf_avg = (output_paf + output_paf_flip[:, ::-1, :][:, :, flip_paf_ord]) / 2  # > (featH, featW, 30)
            output_kp_avg = (output_kp + output_kp_flip[:, ::-1, :][:, :, flip_heat_ord]) / 2  # > (featH, featW, 20)
        else:
            output_paf_avg = output_paf  # > (featH, featW, 30)
            output_kp_avg = output_kp  # > (featH, featW, 20)

        # > `[1-6]`: extract outputs, resize, and remove padding
        # > `heatmap`: (featH, featW, 20) -> (scale_padH, scale_padW, 20)
        heatmap = cv2.resize(output_kp_avg,   # > `KP`: (featH, featW, 20)
                             (0, 0),
                             fx=model_params['stride'],  # > `stride`: 4
                             fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)

        # > `paf`: (featH, featW, 30) -> (scale_padH, scale_padW, 30)
        paf = cv2.resize(output_paf_avg,
                         (0, 0),
                         fx=model_params['stride'],  # > `stride`: 4
                         fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)

        if angle != 0:  # ADDED
            heatmap = cv2.warpAffine(heatmap, rotate_matrix_reverse, (0, 0))
            paf = cv2.warpAffine(paf, rotate_matrix_reverse, (0, 0))

        # `[1-7]`: scale feature maps
        # `pad`: [up, left, down, right]
        # > feat size * stride -> scaled size
        heatmap = heatmap[pad[0]:scale_padH - pad[2], pad[1]:scale_padW - pad[3], :]  # CHANGED-> (scaleH, scaleW, 20)
        paf = paf[pad[0]:scale_padH - pad[2], pad[1]:scale_padW - pad[3], :]  # CHANGED-> (scaleH, scaleW, 30)

        # > scaled size -> img size
        heatmap = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_CUBIC)  # -> (imgH, imgW, 20)
        paf = cv2.resize(paf, (img_w, img_h), interpolation=cv2.INTER_CUBIC)  # -> (imgH, imgW, 30)

        # > `[1-8]-avg`: take the average
        heatmap_avg = heatmap_avg + heatmap / (len(multiplier) * len(rotate_angle))  # -> (imgH, imgW, 20)
        paf_avg = paf_avg + paf / (len(multiplier) * len(rotate_angle))  # -> (imgH, imgW, 30)

        # heatmap_avg = np.maximum(heatmap_avg, heatmap)
        # paf_avg = np.maximum(paf_avg, paf)  # 如果换成取最大，效果会变差，有很多误检

    return heatmap_avg, paf_avg


def find_peaks(heatmap_avg, params):
    all_peaks = []
    peak_counter = 0

    # > `heatmap_avg`: (imgH, imgW, 20)
    heatmap_avg = heatmap_avg.astype(np.float32)

    # > (imgH, imgW, 20) -> (imgH, imgW, 18) -> (18, imgH, imgW) -> (1, 18, imgH, imgW)
    filter_map = heatmap_avg[:, :, :18].copy().transpose((2, 0, 1))[None, ...]
    filter_map = torch.from_numpy(filter_map).cuda()

    # NOTE: Add Gaussian smooth will be BAD
    # smoothing = util.GaussianSmoothing(18, 5, 3)
    # filter_map = F.pad(filter_map, (2, 2, 2, 2), mode='reflect')
    # filter_map = smoothing(filter_map)

    # > (1, 18, imgH, imgW), `thre1`: 0.1
    filter_map = util.keypoint_heatmap_nms(filter_map, kernel=3, thre=params['thre1'])
    filter_map = filter_map.cpu().numpy().squeeze().transpose((1, 2, 0))  # > (imgH, imgW, 18)
    # > `heatmap_avg`: (imgH, imgW, 20)
    for part in range(18):  # > `#kp`: 没有对背景（序号19）取非极大值抑制NMS
        map_orig = heatmap_avg[:, :, part]  # > (imgH, imgW)
        # NOTE: 在某些情况下，需要对一个像素的周围的像素给予更多的重视。因此，可通过分配权重来重新计算这些周围点的值。
        # 这可通过高斯函数（钟形函数，即喇叭形数）的权重方案来解决。
        peaks_binary = filter_map[:, :, part]  # > (imgH, imgW)
        peak_y, peak_x = np.nonzero(peaks_binary)
        peaks = list(zip(peak_x, peak_y))  # > (#peaks, (x,y))
        # > `offset_radius`: 2, `refined_peaks_with_score`: (x,y,score)
        refined_peaks_with_score = [util.refine_centroid(map_orig, anchor, params['offset_radius']) for anchor in peaks]
        # peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in refined_peaks]

        # > TOCHECK: `id`: [0, #peaks), `refined_peaks_with_score`: (#peaks, (x,y,score))
        id = range(peak_counter, peak_counter + len(refined_peaks_with_score))  # `id`: len(x) = #peaks

        # > [(x,y,score) + (id,) = (x,y,score,id)] of `certain type` of keypoint.
        # 为每一个相应peak (parts)都依次编了一个号
        peaks_with_score_and_id = [refined_peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)  # refined_peaks

    return all_peaks  # > (#kp, #peaks, (x,y,score,kp_id)


def find_connections(all_peaks, paf_avg, image_width, params):
    # > `all_peaks`: (#kp, (x,y,score,kp_id)
    connected_limbs = []  # > (#connect, 6=(A_peak_id, B_peak_id, dist_prior, partA_id, partB_id, limb_len))
    special_k = []

    paf_xy_coords_per_limb = np.arange(14).reshape(7, 2)  # > (7, 2)
    limb_intermed_coords = np.empty((4, params['mid_num']), dtype=np.intp)  # > (4, 20)

    # > `#limb` = `#connection` = `#paf_channel`, `limb_pairs[i]`=(kp_id1, kp_id2)
    for pair_id in range(len(joint2limb_pairs)):  # > 30 pairs, 最外层的循环是某一个limb_pairs，因为mapIdx个数与之是一致对应的
        # 某一个channel上limb的响应热图, 它的长宽与原始输入图片大小一致，前面经过resize了
        score_mid = paf_avg[:, :, pair_id]  # (imgH, imgW, 30) -> `c=k` -> (imgH, imgW)
        # score_mid = gaussian_filter(score_mid, sigma=3)

        # `all_peaks(list)`: (#kp, #peaks, (x,y,score,id)), 每一行也是一个list,保存了检测到的特定的parts(joints)
        joints_src = all_peaks[joint2limb_pairs[pair_id][0]]  # > `all_peaks` -> `kp_id1` -> (x,y,score,id)
        # 注意具体处理时标号从0还是1开始。从收集的peaks中取出某类关键点（part)集合
        joints_dst = all_peaks[joint2limb_pairs[pair_id][1]]  # > `all_peaks` -> `kp_id2` -> (x,y,score,id)
        if len(joints_src) == 0 and len(joints_dst) == 0:
            # 一个空的[]也能加入到list中，这一句是必须的！因为connection_all的数据结构是每一行代表一类limb connection
            special_k.append(pair_id)
            connected_limbs.append([])
        else:
            connection_candidates = []
            # map `kp1` to `kp2` of certain pair from person to person
            for i, joint_src in enumerate(joints_src):
                for j, joint_dst in enumerate(joints_dst):
                    joint_src = np.array(joint_src)
                    joint_dst = np.array(joint_dst)
                    # Subtract the position of both joints to obtain the direction of the potential limb
                    limb_dir = joint_dst[:2] - joint_src[:2]  # > (x,y)
                    # Compute the distance/length of the potential limb (norm of limb_dir)
                    # limb_len = np.sqrt(limb_dir[0] * limb_dir[0] + limb_dir[1] * limb_dir[1])  # > `limb_len`: scalar
                    limb_len = np.sqrt(np.sum(limb_dir ** 2)) + 1e-8
                    limb_dir = limb_dir / limb_len

                    mid_num = min(int(round(limb_len + 1)), params['mid_num'])  # > `mid_num`: 20, TOCHECK: `mid_num`?
                    # TOCHECK: failure case when 2 body parts overlaps
                    if limb_len == 0:  # 为了跳过出现不同节点相互覆盖出现在同一个位置，也有说norm加一个接近0的项避免分母为0
                        # SEE：https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/54
                        continue

                    # > TOCHECK: `candA/B[peak_id]`: (x,y,score,id) -> `startend`: (mid_num, (x,y))
                    # > [orig]
                    # startend = list(zip(np.round(np.linspace(start=joint_src[0], stop=joint_dst[0], num=mid_num)),
                    #                     np.round(np.linspace(start=joint_src[1], stop=joint_dst[1], num=mid_num))))
                    # limb_response 是代表某一个limb通道下的heat map响应
                    # > `score_mid` of `paf_avg`: (imgH, imgW) -> `score_mid[y, x]` -> `limb_response`: (mid_num,)
                    # limb_response = np.array([
                    #     score_mid[int(loc[1]), int(loc[0])]
                    #     for loc in startend
                    # ])  # > (imgH, imgW) -> (20,): [0, ..., mid_num-1]

                    # > [new], MY-TODO: re-written into C++
                    limb_intermed_x = np.round(np.linspace(start=joint_src[0], stop=joint_dst[0], num=mid_num)).astype(np.intp)
                    limb_intermed_y = np.round(np.linspace(start=joint_src[1], stop=joint_dst[1], num=mid_num)).astype(np.intp)
                    limb_response = score_mid[limb_intermed_y, limb_intermed_x]  # > (20,)

                    score_midpts = limb_response  # > (mid_num,)
                    # > `score_with_dist_prior`: scalar
                    score_with_dist_prior = score_midpts.mean() + min(0.5 * image_width / limb_len - 1, 0)
                    # 这一项是为了惩罚过长的connection, 只有当长度大于图像高度的一半时才会惩罚 todo
                    # The term of sum(score_midpts)/len(score_midpts), see the link below.
                    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/48
                    # > `thre2`: 0.1, `connect_ration`: 0.8 -> `criterion1`: True/False
                    criterion1 = \
                        np.count_nonzero(score_midpts > params['thre2']) >= mid_num * params['connect_ration']
                    # 我认为这个判别标准是保证paf朝向的一致性  threshold = param['thre2'] =0.12
                    # CMU原始项目中parm['thre2'] = 0.05
                    criterion2 = score_with_dist_prior > 0  # > True/False
                    if criterion1 and criterion2:
                        # > TOCHECK: [0.5, 0.25, 0.25] -> (candA_id, candB_id, dist_prior, limb_len, `confidence`)
                        connection_candidates.append([
                            i, j,  # TOCHECK cand_ID?
                            score_with_dist_prior,  # > scalar
                            limb_len,
                            # TOCHECK: weighted sum?
                            #  CHANGED: can be optimized if weights are removed
                            0.5 * score_with_dist_prior +
                            0.25 * joint_src[2] +
                            0.25 * joint_dst[2]
                        ])
                        # connection_candidate排序的依据是dist prior概率和两个端点heat map预测的概率值
                        # How to understand the criterion?

            # sort by `confidence` -> (#person, (candA_id, candB_id, dist_prior, limb_len, `confidence`)) -> [max, ..., min]
            connection_candidates = sorted(connection_candidates, key=lambda x: x[4], reverse=True)
            # sorted 函数对可迭代对象，按照key参数指定的对象进行排序，revers=True是按照逆序排序, order: big->small
            # DELETED: sort之后可以把最可能是limb的留下，而把和最可能是limb的端点竞争的端点删除
            max_connections = min(len(joints_src), len(joints_dst))
            connections = np.zeros((0, 6))
            for potential_connect in connection_candidates:  # 根据`confidence`的顺序选择connections
                i, j, s, limb_len = potential_connect[0:4]
                if i not in connections[:, 3] and j not in connections[:, 4]:
                    # 进行判断确保不会出现两个端点集合A,B中，出现一个集合中的点与另外一个集合中两个点同时相连
                    connections = np.vstack([
                        connections,
                        # `src_peak_id`, `dst_peak_id`, `dist_prior`, `joint_src_id`, `joint_dst_id`, `limb_len`
                        [joints_src[i][3], joints_dst[j][3], s, i, j, limb_len]
                    ])  # > (#connect, 6)

                    if len(connections) >= max_connections:  # 会出现关节点不够连的情况
                        break
            connected_limbs.append(connections)

    return connected_limbs, special_k


def find_people(connected_limbs, special_k, joint_list, params):
    """
     `connected_limbs`: (#connect, 6=(A_peak_id, B_peak_id, dist_prior, A_part_id, B_part_id, limb_len))
     `connected_limbs[k]`: 保存的是第k个类型的所有limb连接，可能有多个，也可能一个没有
     `connected_limbs` 每一行是一个类型的limb, 每一行格式: N * [idA, idB, score, i, j]
      `subset`: 每一行对应的是 (一个人, 18个关键点, (number, score的结果)) = (#person, 18, 2) + (#person, 2, 2)
     `connected_limbs` 每一行的list保存的是一类limb(connection),遍历所有此类limb,一般的有多少个特定的limb就有多少个人
        # `special_K` ,表示没有找到关节点对匹配的肢体

    """
    # last number in each row is the `total parts number of that person`
    # the second last number in each row is `the score of the overall configuration`
    person_to_joint_assoc = -1 * np.ones((0, 20, 2))  # > (#person, 20, 2)
    # `joint_list` -> `candidate`: (#kp * person, (x,y,score,id))
    candidate = np.array([item for sublist in joint_list for item in sublist])
    # candidate.shape = (94, 4). 列表解析式，两层循环，先从all peaks取，再从sublist中取。 all peaks是两层list

    for limb_type in range(len(joint2limb_pairs)):  # > #pairs
        if limb_type not in special_k:  # 即　有与之相连的，这个paf(limb)是存在的
            joint_src_type, joint_dst_type = joint2limb_pairs[limb_type]

            for limb_id, limb_info in enumerate(connected_limbs[limb_type]):

                found = 0
                found_subset_idx = [-1, -1]  # 每次循环只解决两个part，所以标记只需要两个flag
                for person_id, person_limbs in enumerate(person_to_joint_assoc):  # > #person
                    if person_limbs[joint_src_type][0] == limb_info[0] or \
                       person_limbs[joint_dst_type][0] == limb_info[1]:
                        # check if two joints of a limb is used in previous step, which means it is used by someone.
                        if found >= 2:
                            print('************ error occurs! 3 joints sharing have been found  *******************')
                            continue
                        found_subset_idx[found] = person_id  # 标记一下，这个端点应该是第j个人的
                        found += 1

                if found == 1:
                    person_id = found_subset_idx[0]
                    # > `len_rate`:16
                    if person_to_joint_assoc[person_id][joint_dst_type][0].astype(int) == -1 and \
                       person_to_joint_assoc[person_id][-1][1] * params['len_rate'] > limb_info[-1]:
                        # 如果新加入的limb比之前已经组装的limb长很多，也舍弃
                        # 如果这个人的当前点还没有被找到时，把这个点分配给这个人
                        # 这一个判断非常重要，因为第18和19个limb分别是 2->16, 5->17,这几个点已经在之前的limb中检测到了，
                        # 所以如果两次结果一致，不更改此时的part分配，否则又分配了一次，编号是覆盖了，但是继续运行下面代码，part数目
                        # 会加１，结果造成一个人的part之和>18。不过如果两侧预测limb端点结果不同，还是会出现number of part>18，造成多检
                        # TOCHECK: 没有利用好冗余的connection信息，最后两个limb的端点与之前循环过程中重复了，但没有利用聚合，
                        #  只是直接覆盖，其实直接覆盖是为了弥补漏检

                        person_to_joint_assoc[person_id][joint_dst_type][0] = limb_info[1]  # partBs[i]是limb其中一个端点的id号码
                        person_to_joint_assoc[person_id][joint_dst_type][1] = limb_info[2]  # 保存这个点被留下来的置信度
                        person_to_joint_assoc[person_id][-1][0] += 1  # last number in each row is the total parts number of that person

                        # # subset[j][-2][1]用来记录不包括当前新加入的类型节点时的总体初始置信度，引入它是为了避免下次迭代出现同类型关键点，覆盖时重复相加了置信度
                        # subset[j][-2][1] = subset[j][-2][0]  # 因为是不包括此类节点的初始值，所以只会赋值一次 !!

                        person_to_joint_assoc[person_id][-2][0] += candidate[limb_info[1].astype(int), 2] + limb_info[2]
                        person_to_joint_assoc[person_id][-1][1] = max(limb_info[-1], person_to_joint_assoc[person_id][-1][1])

                        # the second last number in each row is the score of the overall configuration

                    elif person_to_joint_assoc[person_id][joint_dst_type][0].astype(int) != limb_info[1].astype(int):
                        if person_to_joint_assoc[person_id][joint_dst_type][1] >= limb_info[2]:
                            # 如果考察的这个limb连接没有已经存在的可信，则跳过
                            pass

                        else:
                            # 否则用当前的limb端点覆盖已经存在的点，并且在这之前，减去已存在关节点的置信度和连接它的limb置信度
                            if params['len_rate'] * person_to_joint_assoc[person_id][-1][1] <= limb_info[-1]:
                                continue
                            # 减去之前的节点置信度和limb置信度
                            person_to_joint_assoc[person_id][-2][0] -= candidate[person_to_joint_assoc[person_id][joint_dst_type][0].astype(int), 2] + \
                                                       person_to_joint_assoc[person_id][joint_dst_type][1]

                            # 添加当前节点
                            person_to_joint_assoc[person_id][joint_dst_type][0] = limb_info[1]
                            person_to_joint_assoc[person_id][joint_dst_type][1] = limb_info[2]  # 保存这个点被留下来的置信度
                            person_to_joint_assoc[person_id][-2][0] += candidate[limb_info[1].astype(int), 2] + limb_info[2]

                            person_to_joint_assoc[person_id][-1][1] = max(limb_info[-1], person_to_joint_assoc[person_id][-1][1])

                    # overlap the reassigned keypoint with higher score
                    #  如果是添加冗余连接的重复的点，用新的更加高的冗余连接概率取代原来连接的相同的关节点的概率
                    # -- 对上面问题的回答： 使用前500进行测试，发现加上这个能提高0.1%，没有什么区别
                    elif person_to_joint_assoc[person_id][joint_dst_type][0].astype(int) == limb_info[1].astype(int) and \
                         person_to_joint_assoc[person_id][joint_dst_type][1] <= limb_info[2]:
                        # 否则用当前的limb端点覆盖已经存在的点，并且在这之前，减去已存在关节点的置信度和连接它的limb置信度

                        # 减去之前的节点置信度和limb置信度
                        person_to_joint_assoc[person_id][-2][0] -= candidate[person_to_joint_assoc[person_id][joint_dst_type][0].astype(int), 2] + person_to_joint_assoc[person_id][joint_dst_type][1]
                        person_to_joint_assoc[person_id][joint_dst_type][0] = limb_info[1]
                        person_to_joint_assoc[person_id][joint_dst_type][1] = limb_info[2]  # 保存这个点被留下来的置信度
                        person_to_joint_assoc[person_id][-2][0] += candidate[limb_info[1].astype(int), 2] + limb_info[2]

                        person_to_joint_assoc[person_id][-1][1] = max(limb_info[-1], person_to_joint_assoc[person_id][-1][1])

                elif found == 2:  # if found 2 and disjoint, merge them (disjoint：不相交)
                    # -----------------------------------------------------
                    # 如果肢体组成的关节点A,B分别连到了两个人体，则表明这两个人体应该组成一个人体，
                    # 则合并两个人体（当肢体是按顺序拼接情况下不存在这样的状况）
                    # --------------------------------------------------

                    # 说明组装的过程中，有断掉的情况（有limb或者说connection缺失），在之前重复开辟了一个sub person,其实他们是同一个人上的
                    # If humans H1 and H2 share a part index with the same coordinates, they are sharing the same part!
                    #  H1 and H2 are, therefore, the same humans. So we merge both sets into H1 and remove H2.
                    # https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8
                    # 该代码与链接中的做法有差异，个人认为链接中的更加合理而且更容易理解
                    j1, j2 = found_subset_idx

                    membership1 = ((person_to_joint_assoc[j1][..., 0] >= 0).astype(int))[:-2]  # 用[:,0]也可
                    membership2 = ((person_to_joint_assoc[j2][..., 0] >= 0).astype(int))[:-2]
                    membership = membership1 + membership2
                    # [:-2]不包括最后个数项与scores项
                    # 这些点应该属于同一个人,将这个人所有类型关键点（端点part)个数逐个相加
                    if len(np.nonzero(membership == 2)[0]) == 0:  # if found 2 and disjoint, merge them

                        min_limb1 = np.min(person_to_joint_assoc[j1, :-2, 1][membership1 == 1])
                        min_limb2 = np.min(person_to_joint_assoc[j2, :-2, 1][membership2 == 1])
                        min_tolerance = min(min_limb1, min_limb2)  # 计算允许进行拼接的最低置信度

                        if limb_info[2] < params['connection_tole'] * min_tolerance or params['len_rate'] * \
                                person_to_joint_assoc[j1][-1][1] <= limb_info[-1]:
                            # 如果merge这两个身体部分的置信度不够大，或者当前这个limb明显大于已存在的limb的长度，则不进行连接
                            # todo: finetune the tolerance of connection
                            continue  #

                        person_to_joint_assoc[j1][:-2][...] += (person_to_joint_assoc[j2][:-2][...] + 1)
                        # 对于没有节点标记的地方，因为两行subset相应位置处都是-1,所以合并之后没有节点的部分依旧是-１
                        # 把不相交的两个subset[j1],[j2]中的id号进行相加，从而完成合并，这里+1是因为默认没有找到关键点初始值是-1

                        person_to_joint_assoc[j1][-2:][:, 0] += person_to_joint_assoc[j2][-2:][:, 0]  # 两行subset的点的个数和总置信度相加

                        person_to_joint_assoc[j1][-2][0] += limb_info[2]
                        person_to_joint_assoc[j1][-1][1] = max(limb_info[-1], person_to_joint_assoc[j1][-1][1])
                        # 注意：　因为是disjoint的两行person_to_joint_assoc点的merge，因此先前存在的节点的置信度之前已经被加过了 !! 这里只需要再加当前考察的limb的置信度
                        person_to_joint_assoc = np.delete(person_to_joint_assoc, j2, 0)

                    else:
                        # 出现了两个人同时竞争一个limb的情况，并且这两个人不是同一个人，通过比较两个人包含此limb的置信度来决定，
                        # 当前limb的节点应该分配给谁，同时把之前的那个与当前节点相连的节点(即partsA[i])从另一个人(subset)的节点集合中删除
                        if limb_info[0] in person_to_joint_assoc[j1, :-2, 0]:
                            c1 = np.where(person_to_joint_assoc[j1, :-2, 0] == limb_info[0])
                            c2 = np.where(person_to_joint_assoc[j2, :-2, 0] == limb_info[1])
                        else:
                            c1 = np.where(person_to_joint_assoc[j1, :-2, 0] == limb_info[1])
                            c2 = np.where(person_to_joint_assoc[j2, :-2, 0] == limb_info[0])

                        # c1, c2分别是当前limb连接到j1人的第c1个关节点，j2人的第c2个关节点
                        c1 = int(c1[0])
                        c2 = int(c2[0])
                        assert c1 != c2, "an candidate keypoint is used twice, shared by two people"

                        # 如果当前考察的limb置信度比已经存在的两个人连接的置信度小，则跳过，否则删除已存在的不可信的连接节点。
                        if limb_info[2] < person_to_joint_assoc[j1][c1][1] and limb_info[2] < person_to_joint_assoc[j2][c2][1]:
                            continue  # the trick here is useful

                        small_j = j1
                        big_j = j2
                        remove_c = c1

                        if person_to_joint_assoc[j1][c1][1] > person_to_joint_assoc[j2][c2][1]:
                            small_j = j2
                            big_j = j1
                            remove_c = c2

                        # 删除和当前limb有连接,并且置信度低的那个人的节点 > TOCHECK:  获取不删除？为了检测更多？
                        if params['remove_recon'] > 0:
                            person_to_joint_assoc[small_j][-2][0] -= candidate[person_to_joint_assoc[small_j][remove_c][0].astype(int), 2] + \
                                                      person_to_joint_assoc[small_j][remove_c][1]
                            person_to_joint_assoc[small_j][remove_c][0] = -1
                            person_to_joint_assoc[small_j][remove_c][1] = -1
                            person_to_joint_assoc[small_j][-1][0] -= 1

                # if find no partA in the person_to_joint_assoc, create a new person_to_joint_assoc
                # 如果肢体组成的关节点A,B没有被连接到某个人体则组成新的人体
                # ------------------------------------------------------------------
                #    1.Sort each possible connection by its score.
                #    2.The connection with the highest score is indeed a final connection.
                #    3.Move to next possible connection. If no parts of this connection have
                #    been assigned to a final connection before, this is a final connection.
                #    第三点是说，如果下一个可能的连接没有与之前的连接有共享端点的话，会被视为最终的连接，加入row
                #    4.Repeat the step 3 until we are done.
                # 说明见：　https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8

                elif not found and limb_type < len(joint2limb_pairs):
                    #  原始的时候是18,因为我加了limb，所以是24,因为真正的limb是0~16，最后两个17,18是额外的不是limb
                    # FIXME: 但是后面画limb的时候没有把鼻子和眼睛耳朵的连线画上，要改进
                    # `connection_all`: (#connect, 6=(A_peak_id, B_peak_id, dist_prior, A_part_id, B_part_id, limb_len))
                    joint_peak_ids, dist_prior, limb_len = limb_info[:2], limb_info[2], limb_info[-1]
                    row = -1 * np.ones((20, 2))  # > (20, (kp_id, dist_prior)) = `-1`
                    row[joint_src_type][0] = limb_info[0]  # > (#k1_type,)
                    row[joint_src_type][1] = dist_prior  # > (#cand_pairs, 6) -> dist_prior
                    row[joint_dst_type][0] = limb_info[1]  # > (#k2_type,)
                    row[joint_dst_type][1] = dist_prior  # > (#cand_pairs, 6) -> dist_prior
                    # > assign to 19 and 20
                    row[-1][0] = 2  # TOCHECK: why assign `2` here? is it `bg` class?
                    row[-1][1] = limb_len  # `limb_len`: 这一位用来记录上轮连接limb时的长度，用来作为下一轮连接的先验知识
                    # > `candidate`: (#kp * person, (x,y,score,id)) -> peak_ids -> score
                    row[-2][0] = sum(candidate[joint_peak_ids.astype(int), 2]) + dist_prior  # > TOCHECK: `reverse_kp`?
                    # 两个端点的置信度+limb连接的置信度
                    # print('create a new subset:  ', row, '\t')
                    row = row[np.newaxis, :, :]  # 为了进行concatenate，需要插入一个轴, -> (1, 20, 2)
                    person_to_joint_assoc = np.concatenate((person_to_joint_assoc, row), axis=0)  # -> (1, 20, 2) TOCHECK: line 384?
    # 将没有被分配到一些人身上的点分配给距离它们近，并且缺少此类节点的人身上？或许这样做坏处更多
    # Delete people who have very few parts connected
    people_to_delete = []
    for limb_id, person_info in enumerate(person_to_joint_assoc):  # > #person
        # > TOCHECK: `[0]` means ?
        if person_info[-1][0] < 2 or person_info[-2][0] / person_info[-1][0] < 0.45:
            people_to_delete.append(limb_id)
    person_to_joint_assoc = np.delete(person_to_joint_assoc, people_to_delete, axis=0)

    return person_to_joint_assoc, candidate


def get_image_name(coco, image_id):
    return coco.imgs[image_id]['file_name']


def predict_many(coco, images_directory, validation_ids, params, model, model_params, heat_layers, paf_layers):
    assert (not set(validation_ids).difference(set(coco.getImgIds())))

    keypoints = {}

    for image_id in tqdm.tqdm(validation_ids):
        image_name = get_image_name(coco, image_id)
        image_name = os.path.join(images_directory, image_name)
        keypoints[image_id] = process(image_name, dict(params), model, dict(model_params), heat_layers + 2, paf_layers)
        # fixme: heat_layers + 1 if you use background keypoint  !!!
    return keypoints


def format_results(keypoints, resFile):
    format_keypoints = []
    # Question: do we need to sort the detections by scores before evaluation ?
    # -- I think we do not have. COCO will select the top 20 automatically
    for image_id, people in keypoints.items():
        for keypoint_list, score in people:
            format_keypoint_list = []
            for x, y in keypoint_list:
                for v in [x, y, 1 if x > 0 or y > 0 else 0]:  # int(x), int(y)
                    # 坐标取了整数,为了减少文件的大小，如果x,y有一个有值，那么标记这个点为可见。　如果x or y =0,令v=0,coco只评测v>0的点
                    format_keypoint_list.append(v)

            format_keypoints.append({
                "image_id": image_id,
                "category_id": 1,
                "keypoints": format_keypoint_list,
                "score": score,
            })

    json.dump(format_keypoints, open(resFile, 'w'))


def validation(model, dump_name, validation_ids=None, dataset='val2017'):
    annType = 'keypoints'
    prefix = 'person_keypoints'

    dataDir = 'data/dataset/coco'

    # # # #############################################################################
    # For evaluation on validation set
    annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataset)
    print(annFile)
    cocoGt = COCO(annFile)

    if validation_ids == None:  # todo: we can set the validataion image ids here  !!!!!!
        validation_ids = cocoGt.getImgIds()[:500]  # [:1000] we can change the range of COCO validation images here
    # # #############################################################################

    # #############################################################################
    # For evaluation on test-dev set
    # annFile = 'data/dataset/coco/link2coco2017/annotations_trainval_info/image_info_test-dev2017.json' # image_info_test2017.json
    # cocoGt = COCO(annFile)
    # validation_ids = cocoGt.getImgIds()
    # #############################################################################

    resFile = '%s/results/%s_%s_%s100_results.json'
    resFile = resFile % (dataDir, prefix, dataset, dump_name)
    print('the path of detected keypoint file is: ', resFile)
    os.makedirs(os.path.dirname(resFile), exist_ok=True)

    keypoints = predict_many(cocoGt, os.path.join(dataDir, dataset), validation_ids, params, model, model_params,
                             config.heat_layers, config.paf_layers)
    format_results(keypoints, resFile)
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = validation_ids
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
    posenet.load_state_dict(checkpoint['weights'])  # 加入他人训练的模型，可能需要忽略部分层，则strict=False
    print('Network weights have been resumed from checkpoint...')

    if torch.cuda.is_available():
        posenet.cuda()

    posenet = amp.initialize(posenet,
                             opt_level=args.opt_level,
                             keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                             loss_scale=args.loss_scale)
    posenet.eval()  # set eval mode is important

    params, model_params = config_reader()

    # show keypoint assignment algorithm speed
    show_eval_speed = False

    with torch.no_grad():
        eval_result_original = validation(posenet,
                                          dump_name='residual_4_hourglass_focal_epoch_52_512_input_1scale_max',
                                          dataset='val2017')  # 'val2017'


    #  若是在test数据集上进行预测并写结果，则
    # annFile='keras_Realtime_Multi-Person_Pose_Estimation-new-generation/dataset/coco/link2coco2017/annotations_trainval_info/image_info_test2017.json'
    # cocoGt = COCO(annFile)
    # validation_ids = cocoGt.getImgIds() 将获得带有image id的一个list
