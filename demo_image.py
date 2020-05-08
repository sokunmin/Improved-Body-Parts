"""
Hint: please ingore the chinease annotations whcih may be wrong and they are just remains from old version.
"""

import argparse
import math
import os
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from config.config import GetConfig, TrainingOpt
from models.posenet import NetworkEval
from utils import util
from utils.config_reader import config_reader

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

joint2limb_pairs = config.limbs_conn
dt_gt_mapping = config.dt_gt_mapping
flip_heat_ord = config.flip_heat_ord
flip_paf_ord = config.flip_paf_ord
draw_list = config.draw_list


# ###############################################################################################################
def process(input_image_path, params, model_params, heat_layers, paf_layers):
    oriImg = cv2.imread(input_image)  # B,G,R order.    訓練數據的讀入也是用opencv，因此也是B, G, R順序
    img_h, img_w, _ = oriImg.shape
    heatmaps, pafs = predict(oriImg, params, model_params, heat_layers, paf_layers, input_image_path)
    show_color_vector(oriImg, pafs, heatmaps)
    joint_list = find_peaks(heatmaps)
    connection_limbs, special_k = find_connections(joint_list, pafs, img_w)
    person_to_joint_assoc, joint_candidates = find_humans(connection_limbs, joint_list, special_k)

    canvas = cv2.imread(input_image)  # B,G,R order
    # canvas = oriImg
    keypoints = []

    for s in person_to_joint_assoc[..., 0]:
        keypoint_indexes = s[:18]  # 定義的keypoint一共有18個
        person_keypoint_coordinates = []
        for index in keypoint_indexes:
            if index == -1:
                # "No candidate for keypoint" # 標誌為-1的part是沒有檢測到的
                X, Y = 0, 0
            else:
                X, Y = joint_candidates[index.astype(int)][:2]
            person_keypoint_coordinates.append((X, Y))
        person_keypoint_coordinates_coco = [None] * 17

        for dt_index, gt_index in dt_gt_mapping.items():
            if gt_index is None:
                continue
            person_keypoint_coordinates_coco[gt_index] = person_keypoint_coordinates[dt_index]

        keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[-2]))  # s[19] is the score

    for i in range(len(keypoints)):
        print('the {}th keypoint detection result is : '.format(i), keypoints[i])

    # 畫所有的骨架
    color_board = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    color_idx = 0
    for i in draw_list:  # 畫出18個limb　Fixme：我設計了25個limb,畫的limb順序需要調整，相應color數也要增加
        for n in range(len(person_to_joint_assoc)):
            index = person_to_joint_assoc[n][np.array(joint2limb_pairs[i])][..., 0]
            if -1 in index:  # 有-1說明沒有對應的關節點與之相連,即有一個類型的part沒有缺失，無法連接成limb
                continue
            # 在上一個cell中有　canvas = cv2.imread(test_image) # B,G,R order
            cur_canvas = canvas.copy()
            Y = joint_candidates[index.astype(int), 0]
            X = joint_candidates[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 3), int(angle), 0,
                                       360, 1)

            cv2.circle(cur_canvas, (int(Y[0]), int(X[0])), 4, color=[0, 0, 0], thickness=2)
            cv2.circle(cur_canvas, (int(Y[1]), int(X[1])), 4, color=[0, 0, 0], thickness=2)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[color_board[color_idx]])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        color_idx += 1
    return canvas
    return canvas


def predict(oriImg, params, model_params, heat_layers, paf_layers, input_image_path):
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]  # 按照圖片高度進行縮放

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], heat_layers))  # fixme if you change the number of keypoints
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], paf_layers))

    max_downsample = model_params['max_downsample']
    pad_value = model_params['padValue']
    stride = model_params['stride']

    for m in range(len(multiplier)):
        scale = multiplier[m]

        if scale * oriImg.shape[0] > 2300 or scale * oriImg.shape[1] > 3200:
            scale = min(2300 / oriImg.shape[0], 3200 / oriImg.shape[1])
            print("Input image is too big, shrink it !")

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)  # cv2.INTER_CUBIC
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest,
                                                          max_downsample,
                                                          pad_value)

        input_img = np.float32(imageToTest_padded / 255)
        swap_image = input_img[:, ::-1, :].copy()
        input_img = np.concatenate((input_img[None, ...], swap_image[None, ...]), axis=0)  # (2, height, width, channels)
        input_img = torch.from_numpy(input_img).cuda()
        output_tuple = posenet(input_img)

        # ############ different scales can be shown #############
        output = output_tuple[-1][0].cpu().numpy()

        output_blob = output[0].transpose((1, 2, 0))
        output_paf = output_blob[:, :, :config.paf_layers]
        output_heatmap = output_blob[:, :, config.paf_layers:config.num_layers]

        output_blob_flip = output[1].transpose((1, 2, 0))
        output_paf_flip = output_blob_flip[:, :, :config.paf_layers]  # paf layers
        output_heatmap_flip = output_blob_flip[:, :, config.paf_layers:config.num_layers]  # keypoint layers

        # ################################## flip ensemble ################################
        output_paf_avg = (output_paf + output_paf_flip[:, ::-1, :][:, :, flip_paf_ord]) / 2
        output_heatmap_avg = (output_heatmap + output_heatmap_flip[:, ::-1, :][:, :, flip_heat_ord]) / 2

        # extract outputs, resize, and remove padding
        heatmap = cv2.resize(output_heatmap_avg, (0, 0),
                             fx=stride,
                             fy=stride,
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        # output_blob0 is PAFs
        paf = cv2.resize(output_paf_avg, (0, 0),
                         fx=stride,
                         fy=stride,
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)
        # > DELETED
        heatmap_avg[np.isnan(heatmap_avg)] = 0
        paf_avg[np.isnan(paf_avg)] = 0

    return heatmap_avg, paf_avg


def find_peaks(heatmap_avg):
    joint_list = []
    peak_counter = 0

    heatmap_avg = heatmap_avg.astype(np.float32)

    filter_map = heatmap_avg[:, :, :18].copy().transpose((2, 0, 1))[None, ...]
    filter_map = torch.from_numpy(filter_map).cuda()

    filter_map = util.keypoint_heatmap_nms(filter_map, kernel=3, thre=params['thre1'])
    filter_map = filter_map.cpu().numpy().squeeze().transpose((1, 2, 0))

    for part in range(18):  # 沒有對背景（序號19）取非極大值抑制NMS
        map_ori = heatmap_avg[:, :, part]
        peaks_binary = filter_map[:, :, part]

        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        refined_peaks_with_score = [util.refine_centroid(map_ori, anchor, params['offset_radius']) for anchor in peaks]

        id = range(peak_counter, peak_counter + len(refined_peaks_with_score))
        peaks_with_score_and_id = [refined_peaks_with_score[i] + (id[i],) for i in range(len(id))]

        joint_list.append(peaks_with_score_and_id)
        peak_counter += len(peaks)  # refined_peaks

    return joint_list


def find_connections(joint_list, paf_avg, img_width):

    connection_limbs = []
    special_k = []

    for pair_id in range(len(joint2limb_pairs)):
        score_mid = paf_avg[:, :, pair_id]
        joints_src = joint_list[joint2limb_pairs[pair_id][0]]
        joints_dst = joint_list[joint2limb_pairs[pair_id][1]]

        if len(joints_src) == 0 and len(joints_dst) == 0:
            special_k.append(pair_id)
            connection_limbs.append([])
        else:
            connection_candidate = []
            for i, joint_src in enumerate(joints_src):
                for j, joint_dst in enumerate(joints_dst):
                    joint_src = np.array(joint_src)
                    joint_dst = np.array(joint_dst)
                    limb_dir = joint_dst[:2] - joint_src[:2]  # > (x,y)
                    limb_len = np.sqrt(np.sum(limb_dir ** 2)) + 1e-8
                    mid_num = min(int(round(limb_len + 1)), params['mid_num'])
                    if limb_len == 0:  # 為了跳過出現不同節點相互覆蓋出現在同一個位置，也有說norm加一個接近0的項避免分母為0,詳見：
                        # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/54
                        continue

                    limb_intermed_x = np.round(np.linspace(start=joint_src[0], stop=joint_dst[0], num=mid_num)).astype(
                        np.intp)
                    limb_intermed_y = np.round(np.linspace(start=joint_src[1], stop=joint_dst[1], num=mid_num)).astype(
                        np.intp)
                    limb_response = score_mid[limb_intermed_y, limb_intermed_x]  # > (20,)

                    score_midpts = limb_response

                    connect_score = sum(score_midpts) / len(score_midpts) + min(0.5 * img_width / limb_len - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > params['connect_ration'] * len(
                        score_midpts)  # fixme: tune 手動調整, 本來是 > 0.8*len
                    criterion2 = connect_score > 0

                    if criterion1 and criterion2:
                        connection_candidate.append([
                            i, j, connect_score, limb_len,
                            0.5 * connect_score +
                            0.25 * joint_src[2] +
                            0.25 * joint_dst[2]
                        ])

            max_connections = min(len(joints_src), len(joints_dst))
            connection_candidate = sorted(connection_candidate, key=lambda x: x[4], reverse=True)
            connection = np.zeros((0, 6))
            for c in range(len(connection_candidate)):  # 根據confidence的順序選擇connections
                i, j, s, limb_len = connection_candidate[c][0:4]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([
                        connection,
                        [joints_src[i][3], joints_dst[j][3], s, i, j, limb_len]
                    ])  # 後面會被使用
                    if len(connection) >= max_connections:  # 會出現關節點不夠連的情況
                        break
            connection_limbs.append(connection)
    return connection_limbs, special_k


def find_humans(connection_limbs, joint_list, special_k):

    person_to_joint_assoc = -1 * np.ones((0, 20, 2))
    joint_candidates = np.array([item for sublist in joint_list for item in sublist])

    len_rate = params['len_rate']
    connection_tole = params['connection_tole']
    remove_recon = params['remove_recon']

    for limb_type in range(len(joint2limb_pairs)):
        if limb_type not in special_k:  # 即　有與之相連的，這個paf(limb)是存在的
            joint_src_type, joint_dst_type = joint2limb_pairs[limb_type]

            for limb_id, limb_info in enumerate(connection_limbs[limb_type]):

                limb_src_peak_id = limb_info[0]
                limb_dst_peak_id = limb_info[1]
                limb_connect_score = limb_info[2]
                limb_len = limb_info[-1]

                person_assoc_idx = []
                for person_id, person_limbs in enumerate(person_to_joint_assoc):
                    if person_limbs[joint_src_type, 0].astype(int) == limb_src_peak_id.astype(int) or \
                       person_limbs[joint_dst_type, 0].astype(int) == limb_dst_peak_id.astype(int):
                        person_assoc_idx.append(person_id)

                if len(person_assoc_idx) == 1:

                    person_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                    person_dst_peak_id = person_limbs[joint_dst_type, 0]
                    person_dst_connect_score = person_limbs[joint_dst_type, 1]
                    person_limb_len = person_limbs[-1, 1]

                    # dst joint is connected yet.
                    if person_dst_peak_id.astype(int) == -1 and len_rate * person_limb_len > limb_len:
                        person_limbs[joint_dst_type] = [limb_dst_peak_id, limb_connect_score]
                        person_limbs[-1][0] += 1  # last number in each row is the total parts number of that person
                        person_limbs[-2][0] += joint_candidates[limb_dst_peak_id.astype(int), 2] + limb_connect_score
                        person_limbs[-1][1] = max(limb_len, person_limb_len)
                        # the second last number in each row is the score of the overall configuration

                    # dst joint is connected, but peak_id is not same.
                    elif person_dst_peak_id.astype(int) != limb_dst_peak_id.astype(int):
                        # prev score is higher than current one.
                        if person_dst_connect_score >= limb_connect_score:
                            pass
                        else:  # prev score is lower than current one
                            # TOCHECK: prev limb x 16 is shorter than current limb, then do nothing.
                            if len_rate * person_limb_len <= limb_len:
                                continue

                            person_limbs[-2][0] -= joint_candidates[person_dst_peak_id.astype(int), 2] + person_dst_connect_score
                            person_limbs[joint_dst_type] = [limb_dst_peak_id, limb_connect_score]
                            person_limbs[-2][0] += joint_candidates[limb_dst_peak_id.astype(int), 2] + limb_connect_score

                            person_limbs[-1][1] = max(limb_len, person_limb_len)

                    # dst joint is connected, and peak_id is same, but prev score is lower than current one.
                    elif person_dst_peak_id.astype(int) == limb_dst_peak_id.astype(int) and \
                         person_dst_connect_score <= limb_connect_score:
                        # TOCHECK: why compare length?
                        if len_rate * person_limb_len <= limb_len:
                            continue
                        person_limbs[-2][0] -= joint_candidates[person_dst_peak_id.astype(int), 2] + person_dst_connect_score
                        person_limbs[joint_dst_type] = [limb_dst_peak_id, limb_connect_score]
                        person_limbs[-2][0] += joint_candidates[limb_dst_peak_id.astype(int), 2] + limb_connect_score

                        person_limbs[-1][1] = max(limb_len, person_limb_len)

                elif len(person_assoc_idx) == 2:  # if found 2 and disjoint, merge them (disjoint：不相交)
                    person1_id, person2_id = person_assoc_idx[0], person_assoc_idx[1]
                    person1_limbs = person_to_joint_assoc[person1_id]  # > (20,2)
                    person2_limbs = person_to_joint_assoc[person2_id]  # > (20,2)

                    membership1 = ((person1_limbs[..., 0] >= 0).astype(int))[:-2]  # 用[:,0]也可
                    membership2 = ((person2_limbs[..., 0] >= 0).astype(int))[:-2]
                    membership = membership1 + membership2
                    if len(np.nonzero(membership == 2)[0]) == 0:  # if found 2 and disjoint, merge them

                        min_limb1 = np.min(person1_limbs[:-2, 1][membership1 == 1])
                        min_limb2 = np.min(person2_limbs[:-2, 1][membership2 == 1])
                        min_tolerance = min(min_limb1, min_limb2)

                        if limb_connect_score < connection_tole * min_tolerance or \
                                len_rate * person1_limbs[-1][1] <= limb_len:
                            continue  #

                        person1_limbs[:-2][...] += (person2_limbs[:-2][...] + 1)
                        person1_limbs[-2:][:, 0] += person2_limbs[-2:][:, 0]
                        person1_limbs[-2][0] += limb_connect_score
                        person1_limbs[-1][1] = max(limb_len, person1_limbs[-1, 1])
                        person_to_joint_assoc = np.delete(person_to_joint_assoc, person2_id, 0)

                    else:
                        if limb_src_peak_id in person1_limbs[:-2, 0]:
                            c1 = np.where(person1_limbs[:-2, 0] == limb_src_peak_id)
                            c2 = np.where(person2_limbs[:-2, 0] == limb_dst_peak_id)
                        else:
                            c1 = np.where(person1_limbs[:-2, 0] == limb_dst_peak_id)
                            c2 = np.where(person2_limbs[:-2, 0] == limb_src_peak_id)

                        c1 = int(c1[0])
                        c2 = int(c2[0])
                        assert c1 != c2, "an candidate keypoint is used twice, shared by two people"

                        if limb_connect_score < person1_limbs[c1, 1] and \
                            limb_connect_score < person2_limbs[c2, 1]:
                            continue  # the trick here is useful

                        small_j = person1_id
                        big_j = person2_id
                        remove_c = c1

                        if person1_limbs[c1, 1] > person2_limbs[c2, 1]:
                            small_j = person2_id
                            big_j = person1_id
                            remove_c = c2
                        if remove_recon > 0:
                            person_to_joint_assoc[small_j, -2, 0] -= \
                                joint_candidates[person_to_joint_assoc[small_j, remove_c, 0].astype(int), 2] + \
                                person_to_joint_assoc[small_j, remove_c, 1]
                            person_to_joint_assoc[small_j, remove_c, 0] = -1
                            person_to_joint_assoc[small_j, remove_c, 1] = -1
                            person_to_joint_assoc[small_j, -1, 0] -= 1

                elif limb_type < len(joint2limb_pairs):
                    row = -1 * np.ones((20, 2))
                    row[joint_src_type] = [limb_src_peak_id, limb_connect_score]
                    row[joint_dst_type] = [limb_dst_peak_id, limb_connect_score]
                    row[-1] = [2, limb_len]
                    row[-2][0] = sum(joint_candidates[limb_info[:2].astype(int), 2]) + limb_connect_score
                    row = row[np.newaxis, :, :]  # 為了進行concatenate，需要插入一個軸
                    person_to_joint_assoc = np.concatenate((person_to_joint_assoc, row), axis=0)

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for limb_id in range(len(person_to_joint_assoc)):
        # CHANGED: 4 -> 2
        if person_to_joint_assoc[limb_id][-1][0] < 4 or person_to_joint_assoc[limb_id][-2][0] / person_to_joint_assoc[limb_id][-1][0] < 0.45:
            deleteIdx.append(limb_id)
    person_to_joint_assoc = np.delete(person_to_joint_assoc, deleteIdx, axis=0)

    return person_to_joint_assoc, joint_candidates


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

    print('Resuming from checkpoint ...... ')
    checkpoint = torch.load(opt.ckpt_path, map_location=torch.device('cpu'))  # map to cpu to save the gpu memory
    posenet.load_state_dict(checkpoint['weights'])  # 加入他人訓練的模型，可能需要忽略部分層，則strict=False
    print('Network weights have been resumed from checkpoint...')

    if torch.cuda.is_available():
        posenet.cuda()

    from apex import amp

    posenet = amp.initialize(posenet,
                             opt_level=args.opt_level,
                             keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                             loss_scale=args.loss_scale)
    posenet.eval()  # set eval mode is important

    tic = time.time()
    print('start processing...')
    # load config
    params, model_params = config_reader()
    tic = time.time()
    # generate image with body parts
    with torch.no_grad():
        canvas = process(input_image, params, model_params, config.heat_layers + 2,
                         config.paf_layers)  # todo background + 2

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    plt.imshow(canvas[:, :, [2,1,0]])
    plt.show()
