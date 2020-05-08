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
    person_to_joint_assoc, joint_candidates = postprocess(connection_limbs, joint_list, special_k)

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

    for k in range(len(joint2limb_pairs)):  # 最外層的循環是某一個limbSeq
        score_mid = paf_avg[:, :, k]  # 某一個channel上limb的響應熱圖, 它的長寬與原始輸入圖片大小一致，前面經過resize了
        candA = joint_list[joint2limb_pairs[k][0]]  # all_peaks是list,每一行也是一個list,保存了檢測到的特定的parts(joints)
        candB = joint_list[joint2limb_pairs[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = joint2limb_pairs[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    mid_num = min(int(round(norm + 1)), params['mid_num'])
                    if norm == 0:  # 為了跳過出現不同節點相互覆蓋出現在同一個位置，也有說norm加一個接近0的項避免分母為0,詳見：
                        # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/54
                        continue

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    limb_response = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0]))] \
                                              for I in range(len(startend))])

                    score_midpts = limb_response

                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * img_width / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > params['connect_ration'] * len(
                        score_midpts)  # fixme: tune 手動調整, 本來是 > 0.8*len
                    criterion2 = score_with_dist_prior > 0

                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, norm,
                                                     0.5 * score_with_dist_prior + 0.25 * candA[i][2] + 0.25 * candB[j][
                                                         2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[4], reverse=True)
            connection = np.zeros((0, 6))
            for c in range(len(connection_candidate)):  # 根據confidence的順序選擇connections
                i, j, s, limb_len = connection_candidate[c][0:4]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j, limb_len]])  # 後面會被使用
                    if (len(connection) >= min(nA, nB)):  # 會出現關節點不夠連的情況
                        break
            connection_limbs.append(connection)
        else:
            special_k.append(k)
            connection_limbs.append([])
    return connection_limbs, special_k


def postprocess(connection_limbs, joint_list, special_k):
    # --------------------------------------------------------------------------------------- #
    # ####################################################################################### #
    # --------------------------------- find people ------------------------------------------#
    # ####################################################################################### #
    # --------------------------------------------------------------------------------------- #

    person_to_joint_assoc = -1 * np.ones((0, 20, 2))
    joint_candidates = np.array([item for sublist in joint_list for item in sublist])

    for k in range(len(joint2limb_pairs)):
        if k not in special_k:  # 即　有與之相連的，這個paf(limb)是存在的
            partAs = connection_limbs[k][:, 0]  # limb端點part的序號，也就是保存在candidate中的  id號
            partBs = connection_limbs[k][:, 1]  # limb端點part的序號，也就是保存在candidate中的  id號
            indexA, indexB = np.array(joint2limb_pairs[k])  # 此時處理limb k,limbSeq的兩個端點parts，是parts的類別號.

            for i in range(len(connection_limbs[k])):  # 該層循環是分配k類型的limb connection　(partAs[i],partBs[i])到某個人　subset[]

                found = 0
                subset_idx = [-1, -1]  # 每次循環只解決兩個part，所以標記只需要兩個flag
                for j in range(len(person_to_joint_assoc)):
                    if person_to_joint_assoc[j][indexA][0].astype(int) == (partAs[i]).astype(int) or person_to_joint_assoc[j][indexB][0].astype(
                            int) == partBs[i].astype(int):
                        subset_idx[found] = j  # 標記一下，這個端點應該是第j個人的
                        found += 1

                if found == 1:
                    j = subset_idx[0]

                    if person_to_joint_assoc[j][indexB][0].astype(int) == -1 and \
                            params['len_rate'] * person_to_joint_assoc[j][-1][1] > connection_limbs[k][i][-1]:
                        person_to_joint_assoc[j][indexB][0] = partBs[i]  # partBs[i]是limb其中一個端點的id號碼
                        person_to_joint_assoc[j][indexB][1] = connection_limbs[k][i][2]  # 保存這個點被留下來的置信度
                        person_to_joint_assoc[j][-1][0] += 1  # last number in each row is the total parts number of that person

                        person_to_joint_assoc[j][-2][0] += joint_candidates[partBs[i].astype(int), 2] + connection_limbs[k][i][2]
                        person_to_joint_assoc[j][-1][1] = max(connection_limbs[k][i][-1], person_to_joint_assoc[j][-1][1])
                        # the second last number in each row is the score of the overall configuration

                    elif person_to_joint_assoc[j][indexB][0].astype(int) != partBs[i].astype(int):
                        if person_to_joint_assoc[j][indexB][1] >= connection_limbs[k][i][2]:
                            pass

                        else:
                            if params['len_rate'] * person_to_joint_assoc[j][-1][1] <= connection_limbs[k][i][-1]:
                                continue
                            person_to_joint_assoc[j][-2][0] -= joint_candidates[person_to_joint_assoc[j][indexB][0].astype(int), 2] + person_to_joint_assoc[j][indexB][1]

                            person_to_joint_assoc[j][indexB][0] = partBs[i]
                            person_to_joint_assoc[j][indexB][1] = connection_limbs[k][i][2]  # 保存這個點被留下來的置信度
                            person_to_joint_assoc[j][-2][0] += joint_candidates[partBs[i].astype(int), 2] + connection_limbs[k][i][2]

                            person_to_joint_assoc[j][-1][1] = max(connection_limbs[k][i][-1], person_to_joint_assoc[j][-1][1])

                    elif person_to_joint_assoc[j][indexB][0].astype(int) == partBs[i].astype(int) and person_to_joint_assoc[j][indexB][1] <= \
                            connection_limbs[k][i][2]:
                        if params['len_rate'] * person_to_joint_assoc[j][-1][1] <= connection_limbs[k][i][-1]:
                            continue
                        person_to_joint_assoc[j][-2][0] -= joint_candidates[person_to_joint_assoc[j][indexB][0].astype(int), 2] + person_to_joint_assoc[j][indexB][1]
                        person_to_joint_assoc[j][indexB][0] = partBs[i]
                        person_to_joint_assoc[j][indexB][1] = connection_limbs[k][i][2]  # 保存這個點被留下來的置信度
                        person_to_joint_assoc[j][-2][0] += joint_candidates[partBs[i].astype(int), 2] + connection_limbs[k][i][2]

                        person_to_joint_assoc[j][-1][1] = max(connection_limbs[k][i][-1], person_to_joint_assoc[j][-1][1])

                elif found == 2:  # if found 2 and disjoint, merge them (disjoint：不相交)
                    j1, j2 = subset_idx

                    membership1 = ((person_to_joint_assoc[j1][..., 0] >= 0).astype(int))[:-2]  # 用[:,0]也可
                    membership2 = ((person_to_joint_assoc[j2][..., 0] >= 0).astype(int))[:-2]
                    membership = membership1 + membership2
                    if len(np.nonzero(membership == 2)[0]) == 0:  # if found 2 and disjoint, merge them

                        min_limb1 = np.min(person_to_joint_assoc[j1, :-2, 1][membership1 == 1])
                        min_limb2 = np.min(person_to_joint_assoc[j2, :-2, 1][membership2 == 1])
                        min_tolerance = min(min_limb1, min_limb2)  # 計算允許進行拼接的置信度

                        if connection_limbs[k][i][2] < params['connection_tole'] * min_tolerance or params['len_rate'] * \
                                person_to_joint_assoc[j1][-1][1] <= connection_limbs[k][i][-1]:
                            continue  #

                        person_to_joint_assoc[j1][:-2][...] += (person_to_joint_assoc[j2][:-2][...] + 1)
                        person_to_joint_assoc[j1][-2:][:, 0] += person_to_joint_assoc[j2][-2:][:, 0]  # 兩行subset的點的個數和總置信度相加
                        person_to_joint_assoc[j1][-2][0] += connection_limbs[k][i][2]
                        person_to_joint_assoc[j1][-1][1] = max(connection_limbs[k][i][-1], person_to_joint_assoc[j1][-1][1])
                        person_to_joint_assoc = np.delete(person_to_joint_assoc, j2, 0)

                    else:
                        if connection_limbs[k][i][0] in person_to_joint_assoc[j1, :-2, 0]:
                            c1 = np.where(person_to_joint_assoc[j1, :-2, 0] == connection_limbs[k][i][0])
                            c2 = np.where(person_to_joint_assoc[j2, :-2, 0] == connection_limbs[k][i][1])
                        else:
                            c1 = np.where(person_to_joint_assoc[j1, :-2, 0] == connection_limbs[k][i][1])
                            c2 = np.where(person_to_joint_assoc[j2, :-2, 0] == connection_limbs[k][i][0])

                        c1 = int(c1[0])
                        c2 = int(c2[0])
                        assert c1 != c2, "an candidate keypoint is used twice, shared by two people"

                        if connection_limbs[k][i][2] < person_to_joint_assoc[j1][c1][1] and connection_limbs[k][i][2] < person_to_joint_assoc[j2][c2][1]:
                            continue  # the trick here is useful

                        small_j = j1
                        big_j = j2
                        remove_c = c1

                        if person_to_joint_assoc[j1][c1][1] > person_to_joint_assoc[j2][c2][1]:
                            small_j = j2
                            big_j = j1
                            remove_c = c2
                        if params['remove_recon'] > 0:
                            person_to_joint_assoc[small_j][-2][0] -= joint_candidates[person_to_joint_assoc[small_j][remove_c][0].astype(int), 2] + \
                                                      person_to_joint_assoc[small_j][remove_c][1]
                            person_to_joint_assoc[small_j][remove_c][0] = -1
                            person_to_joint_assoc[small_j][remove_c][1] = -1
                            person_to_joint_assoc[small_j][-1][0] -= 1

                elif not found and k < len(joint2limb_pairs):
                    row = -1 * np.ones((20, 2))
                    row[indexA][0] = partAs[i]
                    row[indexA][1] = connection_limbs[k][i][2]
                    row[indexB][0] = partBs[i]
                    row[indexB][1] = connection_limbs[k][i][2]
                    row[-1][0] = 2
                    row[-1][1] = connection_limbs[k][i][-1]  # 這一位用來記錄上輪連接limb時的長度，用來作為下一輪連接的先驗知識
                    row[-2][0] = sum(joint_candidates[connection_limbs[k][i, :2].astype(int), 2]) + connection_limbs[k][i][2]
                    # 兩個端點的置信度+limb連接的置信度
                    # print('create a new subset:  ', row, '\t')
                    row = row[np.newaxis, :, :]  # 為了進行concatenate，需要插入一個軸
                    person_to_joint_assoc = np.concatenate((person_to_joint_assoc, row), axis=0)

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(person_to_joint_assoc)):
        if person_to_joint_assoc[i][-1][0] < 4 or person_to_joint_assoc[i][-2][0] / person_to_joint_assoc[i][-1][0] < 0.45:
            deleteIdx.append(i)
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
