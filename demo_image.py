"""
Hint: please ingore the chinease annotations whcih may be wrong and they are just remains from old version.
"""

import sys
import json
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import tqdm
import time
import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.config_reader import config_reader
from utils import util
from config.config import GetConfig, COCOSourceConfig, TrainingOpt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from models.posenet import NetworkEval
import warnings
import os
import argparse
from apex import amp
from skimage import io

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

limb_pairs = config.limbs_conn  # > 30
dt_gt_mapping = config.dt_gt_mapping
flip_heat_ord = config.flip_heat_ord
flip_paf_ord = config.flip_paf_ord
draw_list = config.draw_list


# ###############################################################################################################


def show_color_vector(oriImg, paf_avg, heatmap_avg):
    hsv = np.zeros_like(oriImg)  # > (imgH, imgW, 3)
    hsv[..., 1] = 255  # > (imgH, imgW, (0,255,0))
    # > convert from `cartesian` to `polar`, `paf_avg`: (imgH, imgW, 30) -> c=16 -> (imgH, imgW)
    mag, ang = cv2.cartToPolar(paf_avg[:, :, 16], 1.5 * paf_avg[:, :, 16])  # 设置不同的系数，可以使得显示颜色不同 -> (imgH, imgW)

    # TOCHECK: 将弧度转换为角度，同时OpenCV中的H范围是180(0 - 179)，所以再除以2
    # 完成后将结果赋给HSV的H通道，不同的角度(方向)以不同颜色表示
    # 对于不同方向，产生不同色调
    # hsv[...,0]等价于hsv[:,:,0]
    hsv[..., 0] = ang * 180 / np.pi / 2

    # 将矢量大小标准化到0-255范围。因为OpenCV中V分量对应的取值范围是256
    # 对于同一H、S而言，向量的大小越大，对应颜色越亮
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # 最后，将生成好的HSV图像转换为BGR颜色空间
    limb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(limb_flow, alpha=.5)
    plt.show()

    plt.imshow(oriImg[:, :, [2, 1, 0]])
    plt.imshow(paf_avg[:, :, 11], alpha=.6)
    plt.show()
    # `heatmap_avg`: (imgH, imgW, 20) -> c=19 -> (imgH, imgW)
    plt.imshow(heatmap_avg[:, :, -1])
    plt.imshow(oriImg[:, :, [2, 1, 0]], alpha=0.25)  # show a keypoint
    plt.show()

    # `heatmap_avg`: (imgH, imgW, 20) -> c=18 -> (imgH, imgW)
    plt.imshow(heatmap_avg[:, :, -2])
    plt.imshow(oriImg[:, :, [2, 1, 0]], alpha=0.5)  # show the person mask
    plt.show()

    # `heatmap_avg`: (imgH, imgW, 20) -> c=4 -> (imgH, imgW)
    plt.imshow(oriImg[:, :, [2, 1, 0]])  # show a keypoint
    plt.imshow(heatmap_avg[:, :, 4], alpha=.5)
    plt.show()
    t = 2


def process(input_image, params, model_params, heat_layers, paf_layers):
    oriImg = cv2.imread(input_image)  # B,G,R order.    训练数据的读入也是用opencv，因此也是B, G, R顺序
    img_h, img_w, _ = oriImg.shape
    # oriImg = cv2.resize(oriImg, (768, 768))
    # oriImg = cv2.flip(oriImg, 1) 因为训练时作了flip，所以用这种方式提升并没有作用
    multiplier = [x * model_params['boxsize'] / img_h for x in params['scale_search']]  # 按照图片高度进行缩放
    # multipier = [0.21749408983451538, 0.43498817966903075, 0.6524822695035462, 0.8699763593380615],
    # --------------------------------------------------------------------------------------- #
    # ------------------------  scale feature maps up to image size  -----------------------#
    # --------------------------------------------------------------------------------------- #
    # 首先把输入图像高度变成`368`,然后再做缩放
    heatmap_avg = np.zeros((img_h, img_w, heat_layers))  # > `heatmap_avg`: (imgH, imgW, 20)
    paf_avg = np.zeros((img_h, img_w, paf_layers))  # > `paf_layers`: (imgH, imgW, 30)
    # > scale image to different sizes and then detect via model
    for m in range(len(multiplier)):  # > #scales
        scale = multiplier[m]

        if scale * img_h > 2300 or scale * img_w > 3200:
            scale = min(2300 / img_h, 3200 / img_w)
            print("Input image is too big, shrink it !")
        # > `imageToTest`: (scaleH, scaleW, 3)
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # > `imageToTest_padded`: (scale_padH, scale_padW, 3)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest,
                                                          model_params['max_downsample'],  # > 64
                                                          model_params['padValue'])  # > 128
        scale_padH, scale_padW, _ = imageToTest_padded.shape

        # ----------------------------------------------------------
        # > WARN: `[1-1]`: we use OpenCV to read image`(BGR)` all the time
        # Input Tensor: a batch of images within [0,1], required shape in this project : (1, height, width, channels)
        input_img = np.float32(imageToTest_padded / 255)
        # input_img -= np.array(config.img_mean[::-1])  # Notice: OpenCV uses BGR format, reverse the last axises
        # input_img /= np.array(config.img_std[::-1])

        # ----------------------------------------------------------
        # > `[1-2]` :add flip image
        swap_image = input_img[:, ::-1, :].copy()
        # plt.imshow(swap_image[:, :, [2, 1, 0]])  # Opencv image format: BGR
        # plt.show()
        input_img = np.concatenate((input_img[None, ...], swap_image[None, ...]), axis=0)  # (2, H, W, C)
        input_img = torch.from_numpy(input_img).cuda()

        # > `[1-3]-model`(4,)=(2, 50, featH, featW) x 4, `dtype=float16`
        output_tuple = posenet(input_img)

        # ----------------------------------------------------------
        # > `[1-4]`: scales vary according to input image size.
        output = output_tuple[-1][0].cpu().numpy()  # > `1st-level` of feature maps -> (2, 50, featH, featW)

        output_blob = output[0].transpose((1, 2, 0))  # > (featH, featW, 50)
        output_blob0 = output_blob[:, :, :config.paf_layers]  # > `PAF`:(featH, featW, 30)
        output_blob1 = output_blob[:, :, config.paf_layers:config.num_layers]  # > `KP`:(featH, featW, 20)
        # > flipped image output
        output_blob_flip = output[1].transpose((1, 2, 0))  # > (featH, featW, 50)
        output_blob0_flip = output_blob_flip[:, :, :config.paf_layers]  # `PAF`: (featH, featW, 30)
        output_blob1_flip = output_blob_flip[:, :, config.paf_layers:config.num_layers]  # > `KP`: (featH, featW, 20)

        # ----------------------------------------------------------
        # > `[1-5]`: flip ensemble
        output_blob0_avg = (output_blob0 + output_blob0_flip[:, ::-1, :][:, :, flip_paf_ord]) / 2  # > (featH, featW, 30)
        output_blob1_avg = (output_blob1 + output_blob1_flip[:, ::-1, :][:, :, flip_heat_ord]) / 2  # > (featH, featW, 20)

        # extract outputs, resize, and remove padding
        # > `[1-6]-heatmap`: (featH, featW, 20) -> (scale_padH, scale_padW, 20)
        heatmap = cv2.resize(output_blob1_avg, (0, 0),
                             fx=model_params['stride'],  # > `stride`: 4
                             fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        # -> (scaleH, scaleW, 20)
        heatmap = heatmap[:scale_padH - pad[2], :scale_padW - pad[3], :]
        # -> (imgH, imgW, 20)
        heatmap = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

        # > `output_blob0` is PAFs
        # > `[1-7]-paf`: (featH, featW, 30) -> (scale_padH, scale_padW, 30)
        paf = cv2.resize(output_blob0_avg, (0, 0),
                         fx=model_params['stride'],  # > `stride`: 4
                         fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        # -> (scaleH, scaleW, 30)
        paf = paf[:scale_padH - pad[2], :scale_padW - pad[3], :]
        # -> (imgH, imgW, 30)
        paf = cv2.resize(paf, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

        # ##############################     为了让平均heatmap不那么模糊？     ################################
        # heatmap[heatmap < params['thre1']] = 0
        # paf[paf < params['thre2']] = 0
        # ####################################################################################### #
        # > `[1-8]-avg`: `heatmap_avg`: (imgH, imgW, 20), `paf_avg`: (imgH, imgW, 30)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

        heatmap_avg[np.isnan(heatmap_avg)] = 0
        paf_avg[np.isnan(paf_avg)] = 0

        # heatmap_avg = np.maximum(heatmap_avg, heatmap)
        # paf_avg = np.maximum(paf_avg, paf)  # 如果换成取最大，效果会变差，有很多误检

    all_peaks = []
    peak_counter = 0
    # --------------------------------------------------------------------------------------- #
    # ------------------------  show the limb and foreground channel  ----------------------- #
    # --------------------------------------------------------------------------------------- #

    show_color_vector(oriImg, paf_avg, heatmap_avg)

    # --------------------------------------------------------------------------------------- #
    # ------------------------- find & refine keypoints  ------------------------------------ #
    # --------------------------------------------------------------------------------------- #
    # TOCHECK: how does GaussianSmoothing work?
    # smoothing = util.GaussianSmoothing(18, 5, 1)
    # heatmap_avg_cuda = torch.from_numpy(heatmap_avg.transpose((2, 0, 1))).cuda()[None, ...]
    # > `heatmap_avg`: (imgH, imgW, 20)
    heatmap_avg = heatmap_avg.astype(np.float32)
    # > (imgH, imgW, 20) -> (imgH, imgW, 18) -> (18, imgH, imgW) -> (1, 18, imgH, imgW)
    filter_map = heatmap_avg[:, :, :18].copy().transpose((2, 0, 1))[None, ...]
    filter_map = torch.from_numpy(filter_map).cuda()

    # TOCHECK: #######################   Add Gaussian smooth  #######################
    # smoothing = util.GaussianSmoothing(18, 7, 1)
    # filter_map = F.pad(filter_map, (3, 3, 3, 3), mode='reflect')
    # filter_map = smoothing(filter_map)
    # # ######################################################################
    # > (1, 18, imgH, imgW), `thre1`: 0.1
    filter_map = util.keypoint_heatmap_nms(filter_map, kernel=3, thre=params['thre1'])
    filter_map = filter_map.cpu().numpy().squeeze().transpose((1, 2, 0))  # > (imgH, imgW, 18)
    # > `heatmap_avg`: (imgH, imgW, 20)
    for part in range(18):  # > `#kp`: 没有对背景（序号19）取非极大值抑制NMS
        map_ori = heatmap_avg[:, :, part]  # > (imgH, imgW)
        # map = gaussian_filter(map_ori, sigma=3)  # 没有高斯滤波貌似效果更好？
        # map = map_ori
        # map up 是值
        peaks_binary = filter_map[:, :, part]  # > (imgH, imgW)
        peak_y, peak_x = np.nonzero(peaks_binary)
        peaks = list(zip(peak_x, peak_y))  # > (#peaks, (y,x))
        # note reverse. xy坐标系和图像坐标系
        # `np.nonzero`: Return the indices of the elements that are non-zero
        # 添加加权坐标计算，根据不同类型关键点弥散程度不同选择加权的范围, TOCHECK: `offset_radius`: 2, is this learnable? `refine_centroid()` -> (x,y,score)
        refined_peaks_with_score = [util.refine_centroid(map_ori, anchor, params['offset_radius']) for anchor in peaks]

        # peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]  # 列表解析式，生产的是list  # refined_peaks
        # > TOCHECK: `id`: [0, #peaks), `refined_peaks_with_score`: (#peaks, (x,y,score))
        id = range(peak_counter, peak_counter + len(refined_peaks_with_score))  # TOCHECK: peak_id?
        # > [(x,y,score) + (id,) = (x,y,score,id)] of `certain type` of keypoint.
        peaks_with_score_and_id = [refined_peaks_with_score[i] + (id[i],) for i in range(len(id))]  # >
        # 为每一个相应peak (parts)都依次编了一个号

        all_peaks.append(peaks_with_score_and_id)
        # all_peaks.append 如果此种关节类型没有元素，append一个空的list []，例如all_peaks[19]:
        # [(205, 484, 0.9319216758012772, 25),
        # (595, 484, 0.777797631919384, 26),
        # (343, 490, 0.8145177364349365, 27), ....
        peak_counter += len(peaks)  # refined_peaks

    # --------------------------------------------------------------------------------------- #
    # ----------------------------- find connections -----------------------------------------#
    # --------------------------------------------------------------------------------------- #

    connection_all = []  # > (#connect, 6=(A_peak_id, B_peak_id, dist_prior, partA_id, partB_id, limb_len))
    special_k = []
    # > `#limb` = `#connection` = `#paf_channel`, `limb_pairs[i]`=(kp_id1, kp_id2)
    # 有多少个limb,就有多少个connection,相对应地就有多少个paf channel
    for pair_id in range(len(limb_pairs)):  # > 30 connections, 最外层的循环是某一个limb_pairs
        # 某一个channel上limb的响应热图, 它的长宽与原始输入图片大小一致，前面经过resize了
        score_mid = paf_avg[:, :, pair_id]  # (imgH, imgW, 30) -> `c=k` -> (imgH, imgW)
        # score_mid = gaussian_filter(orginal_score_mid, sigma=3)  # TOCHECK: fixme use gaussisan blure?
        # `all_peaks(list)`: (#kp, #peaks, (x,y,score,id)), 每一行也是一个list,保存了检测到的特定的parts(joints)
        candA = all_peaks[limb_pairs[pair_id][0]]  # > `all_peaks` -> `kp_id1` -> (x,y,score,id)
        # 注意具体处理时标号从0还是1开始。从收集的peaks中取出某类关键点（part)集合
        candB = all_peaks[limb_pairs[pair_id][1]]  # > `all_peaks` -> `kp_id2` -> (x,y,score,id)
        nA = len(candA)  # > #person
        nB = len(candB)  # > #person
        indexA, indexB = limb_pairs[pair_id]  # > (kp_id1, kp_id2)
        if nA != 0 and nB != 0:
            connection_candidate = []
            # TOCHECK: map `kp1` to `kp2` of certain pair from person to person
            for person_idx in range(nA):  # > #person
                for candB_id in range(nB):  # > #person
                    vec = np.subtract(candB[candB_id][:2], candA[person_idx][:2])  # > (x,y)
                    limb_len = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])  # > `limb_len`: scalar
                    mid_num = min(int(round(limb_len + 1)), params['mid_num'])  # > `mid_num`: 20, TOCHECK: `mid_num`?
                    # TOCHECK: failure case when 2 body parts overlaps
                    if limb_len == 0:  # TOCHECK: 为了跳过出现不同节点相互覆盖出现在同一个位置，也有说norm加一个接近0的项避免分母为0
                        # 详见：https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/54
                        continue
                    # > TOCHECK: `candA/B[peak_id]`: (x,y,score,id) -> `startend`: (mid_num, (x,y))
                    startend = list(zip(np.linspace(start=candA[person_idx][0], stop=candB[candB_id][0], num=mid_num),
                                        np.linspace(start=candA[person_idx][1], stop=candB[candB_id][1], num=mid_num)))
                    # > `score_mid` of `paf_avg`: (imgH, imgW) -> `score_mid[y, x]` -> `limb_response`: (mid_num,)
                    limb_response = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0]))]
                                              for I in range(len(startend))])  # > [0, ..., mid_num-1]

                    score_midpts = limb_response  # > (mid_num,)
                    # > TOCHECK: `score_with_dist_prior`: scalar
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * img_h / limb_len - 1, 0)
                    # 这一项是为了惩罚过长的connection, 只有当长度大于图像高度的一半时才会惩罚 todo
                    # The term of sum(score_midpts)/len(score_midpts), see the link below.
                    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/48
                    # > `thre2`: 0.1, `connect_ration`: 0.8 -> `criterion1`: True/False
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > \
                                 len(score_midpts) * params['connect_ration']  # TOCHECK: fixme: tune 手动调整, 本来是 > 0.8*len
                    # 我认为这个判别标准是保证paf朝向的一致性  param['thre2']
                    # parm['thre2'] = 0.05
                    criterion2 = score_with_dist_prior > 0  # > True/False

                    if criterion1 and criterion2:
                        # > TOCHECK: [0.5, 0.25, 0.25] -> (candA_id, candB_id, dist_prior, limb_len, `confidence`)
                        connection_candidate.append([person_idx, candB_id,  # TOCHECK cand_ID?
                                                     score_with_dist_prior,  # > scalar
                                                     limb_len,  # > scalar
                                                     0.5 * score_with_dist_prior +
                                                     0.25 * candA[person_idx][2] +
                                                     0.25 * candB[candB_id][2]])
                        # TOCHECK:直接把两种类型概率相加不合理
                        # connection_candidate排序的依据是dist prior概率和两个端点heat map预测的概率值
                        # How to understand the criterion?
            # sort by `confidence` -> (#person, (candA_id, candB_id, dist_prior, limb_len, `confidence`)) -> [max, ..., min]
            connection_candidate = sorted(connection_candidate, key=lambda x: x[4], reverse=True)
            # sorted 函数对可迭代对象，按照key参数指定的对象进行排序，revers=True是按照逆序排序，sort之后可以把最可能是limb的留下，而把和最可能是limb的端点竞争的端点删除

            connection = np.zeros((0, 6))
            for c in range(len(connection_candidate)):  # 根据`confidence`的顺序选择connections
                person_idx, candB_id, s, limb_len = connection_candidate[c][0:4]
                if person_idx not in connection[:, 3] and candB_id not in connection[:, 4]:
                    # 进行判断确保不会出现两个端点集合A,B中，出现一个集合中的点与另外一个集合中两个点同时相连
                    connection = np.vstack([
                        connection,
                        # > (6,) = [`A_peak_id`, `B_peak_id`, `dist_prior`, `partA_id`, `partB_id`, `limb_len`]
                        [candA[person_idx][3], candB[candB_id][3], s, person_idx, candB_id, limb_len]
                    ])  # 后面会被使用
                    # TOCHECK: candA[i][3], candB[j][3]是part的id编号
                    if len(connection) >= min(nA, nB):  # 会出现关节点不够连的情况
                        break
            connection_all.append(connection)
        else:
            special_k.append(pair_id)
            connection_all.append([])
            # 一个空的[]也能加入到list中，这一句是必须的！因为connection_all的数据结构是每一行代表一类limb connection

    # --------------------------------------------------------------------------------------- #
    # --------------------------------- find people ------------------------------------------#
    # --------------------------------------------------------------------------------------- #

    # last number in each row is the `total parts number of that person`
    # the second last number in each row is `the score of the overall configuration`
    subset = -1 * np.ones((0, 20, 2))
    # `all_peaks` -> `candidate`: (#kp * person, (x,y,score,id))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    # candidate[:, 2] *= 0.5  # TOCHECK: FIXME: change it? part confidence * 0.5
    # candidate.shape = (94, 4). 列表解析式，两层循环，先从all peaks取，再从sublist中取。 all peaks是两层list

    for pair_id in range(len(limb_pairs)):  # > #connect
        # ---------------------------------------------------------
        # 外层循环limb  对应论文中，每一个limb就是一个子集，分limb处理,贪心策略?
        # special_K ,表示没有找到关节点对匹配的肢体
        if pair_id not in special_k:  # 即　有与之相连的，这个paf(limb)是存在的
            # > `connection_all`: (#connect, 6=(A_peak_id, B_peak_id, dist_prior, A_part_id, B_part_id, limb_len))
            partAs = connection_all[pair_id][:, 0]  # TOCHECK: limb端点part的序号，也就是保存在candidate中的  id号
            partBs = connection_all[pair_id][:, 1]  # TOCHECK: limb端点part的序号，也就是保存在candidate中的  id号
            # connection_all 每一行是一个类型的limb,每一行格式: N * [idA, idB, score, i, j]
            indexA, indexB = np.array(limb_pairs[pair_id])  # 此时处理limb k,limb_pairs的两个端点parts，是parts的类别号.
            #  根据limb_pairs列表的顺序依次考察某种类型的limb，从一个关节点到下一个关节点
            # 该层循环是分配k类型的limb connection　(partAs[i],partBs[i])到某个人　subset[]
            for person_idx in range(len(connection_all[pair_id])):  # > TOCHECK: #connect to cand_A?
                # ------------------------------------------------
                # 每一行的list保存的是一类limb(connection),遍历所有此类limb,一般的有多少个特定的limb就有多少个人

                found = 0
                found_subset_idx = [-1, -1]  # 每次循环只解决两个part，所以标记只需要两个flag
                for candB_id in range(len(subset)):  # > #person
                    # ----------------------------------------------
                    # 这一层循环是遍历所有的人

                    # 1:size(subset,1), 若subset.shape=(5,20), 则len(subset)=5，表示有5个人
                    # subset每一行对应的是 (一个人, 18个关键点, (number, score的结果)) = (#person, 18, 2) + (#person, 2, 2)
                    if subset[candB_id][indexA][0].astype(int) == (partAs[person_idx]).astype(int) or \
                       subset[candB_id][indexB][0].astype(int) == (partBs[person_idx]).astype(int):
                        # 看看这次考察的limb两个端点之一是否有一个已经在上一轮中出现过了,即是否已经分配给某人了
                        # 每一个最外层循环都只考虑一个limb，因此处理的时候就只会有两种part,即表示为partAs,partBs
                        found_subset_idx[found] = candB_id  # 标记一下，这个端点应该是第j个人的
                        found += 1
                # > `connectA`: (6,) = [A_peak_id, B_peak_id, dist_prior, partA_id, partB_id, limb_len]
                connectA = connection_all[pair_id][person_idx]
                # =================================================================
                if found == 1:
                    candB_id = found_subset_idx[0]
                    # > `len_rate`:16
                    if subset[candB_id][indexB][0].astype(int) == -1 and \
                       subset[candB_id][-1][1] * params['len_rate'] > connectA[-1]:
                        # 如果新加入的limb比之前已经组装的limb长很多，也舍弃
                        # 如果这个人的当前点还没有被找到时，把这个点分配给这个人
                        # 这一个判断非常重要，因为第18和19个limb分别是 2->16, 5->17,这几个点已经在之前的limb中检测到了，
                        # 所以如果两次结果一致，不更改此时的part分配，否则又分配了一次，编号是覆盖了，但是继续运行下面代码，part数目
                        # 会加１，结果造成一个人的part之和>18。不过如果两侧预测limb端点结果不同，还是会出现number of part>18，造成多检
                        # TOCHECK: FIXME: 没有利用好冗余的connection信息，最后两个limb的端点与之前循环过程中重复了，但没有利用聚合，
                        #  只是直接覆盖，其实直接覆盖是为了弥补漏检

                        subset[candB_id][indexB][0] = partBs[person_idx]  # partBs[i]是limb其中一个端点的id号码
                        subset[candB_id][indexB][1] = connectA[2]  # 保存这个点被留下来的置信度
                        subset[candB_id][-1][0] += 1  # last number in each row is the total parts number of that person

                        # # subset[j][-2][1]用来记录不包括当前新加入的类型节点时的总体初始置信度，引入它是为了避免下次迭代出现同类型关键点，覆盖时重复相加了置信度
                        # subset[j][-2][1] = subset[j][-2][0]  # 因为是不包括此类节点的初始值，所以只会赋值一次 !!

                        subset[candB_id][-2][0] += candidate[partBs[person_idx].astype(int), 2] + connectA[2]
                        # candidate的格式为：  (343, 490, 0.8145177364349365, 27), ....
                        subset[candB_id][-1][1] = max(connectA[-1], subset[candB_id][-1][1])

                        # the second last number in each row is the score of the overall configuration

                    elif subset[candB_id][indexB][0].astype(int) != partBs[person_idx].astype(int):
                        if subset[candB_id][indexB][1] >= connectA[2]:
                            # 如果考察的这个limb连接没有已经存在的可信，则跳过
                            pass

                        else:
                            # 否则用当前的limb端点覆盖已经存在的点，并且在这之前，减去已存在关节点的置信度和连接它的limb置信度
                            if params['len_rate'] * subset[candB_id][-1][1] <= connectA[-1]:
                                continue
                            # 减去之前的节点置信度和limb置信度
                            subset[candB_id][-2][0] -= candidate[subset[candB_id][indexB][0].astype(int), 2] + subset[candB_id][indexB][1]

                            # 添加当前节点
                            subset[candB_id][indexB][0] = partBs[person_idx]
                            subset[candB_id][indexB][1] = connectA[2]  # 保存这个点被留下来的置信度
                            subset[candB_id][-2][0] += candidate[partBs[person_idx].astype(int), 2] + connectA[2]

                            subset[candB_id][-1][1] = max(connectA[-1], subset[candB_id][-1][1])

                    #  overlap the reassigned keypoint
                    #  如果是添加冗余连接的重复的点，用新的更加高的冗余连接概率取代原来连接的相同的关节点的概率
                    # 这一个改动没啥影响
                    elif subset[candB_id][indexB][0].astype(int) == partBs[person_idx].astype(int) and \
                         subset[candB_id][indexB][1] <= connectA[2]:
                        # 否则用当前的limb端点覆盖已经存在的点，并且在这之前，减去已存在关节点的置信度和连接它的limb置信度
                        if params['len_rate'] * subset[candB_id][-1][1] <= connectA[-1]:
                            continue
                        # 减去之前的节点置信度和limb置信度
                        subset[candB_id][-2][0] -= candidate[subset[candB_id][indexB][0].astype(int), 2] + subset[candB_id][indexB][1]

                        # 添加当前节点
                        subset[candB_id][indexB][0] = partBs[person_idx]
                        subset[candB_id][indexB][1] = connectA[2]  # 保存这个点被留下来的置信度
                        subset[candB_id][-2][0] += candidate[partBs[person_idx].astype(int), 2] + connectA[2]

                        subset[candB_id][-1][1] = max(connectA[-1], subset[candB_id][-1][1])

                # =================================================================
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

                    membership1 = ((subset[j1][..., 0] >= 0).astype(int))[:-2]  # 用[:,0]也可
                    membership2 = ((subset[j2][..., 0] >= 0).astype(int))[:-2]
                    membership = membership1 + membership2
                    # [:-2]不包括最后个数项与scores项
                    # 这些点应该属于同一个人,将这个人所有类型关键点（端点part)个数逐个相加
                    if len(np.nonzero(membership == 2)[0]) == 0:  # if found 2 and disjoint, merge them

                        min_limb1 = np.min(subset[j1, :-2, 1][membership1 == 1])
                        min_limb2 = np.min(subset[j2, :-2, 1][membership2 == 1])
                        min_tolerance = min(min_limb1, min_limb2)  # 计算允许进行拼接的置信度

                        if connectA[2] < params['connection_tole'] * min_tolerance or params['len_rate'] * \
                                subset[j1][-1][1] <= connectA[-1]:
                            # 如果merge这两个身体部分的置信度不够大，或者当前这个limb明显大于已存在的limb的长度，则不进行连接
                            # todo: finetune the tolerance of connection
                            continue  #

                        subset[j1][:-2][...] += (subset[j2][:-2][...] + 1)
                        # 对于没有节点标记的地方，因为两行subset相应位置处都是-1,所以合并之后没有节点的部分依旧是-１
                        # 把不相交的两个subset[j1],[j2]中的id号进行相加，从而完成合并，这里+1是因为默认没有找到关键点初始值是-1

                        subset[j1][-2:][:, 0] += subset[j2][-2:][:, 0]  # 两行subset的点的个数和总置信度相加

                        subset[j1][-2][0] += connectA[2]
                        subset[j1][-1][1] = max(connectA[-1], subset[j1][-1][1])
                        # 注意：　因为是disjoint的两行subset点的merge，因此先前存在的节点的置信度之前已经被加过了 !! 这里只需要再加当前考察的limb的置信度
                        subset = np.delete(subset, j2, 0)

                    else:
                        # 出现了两个人同时竞争一个limb的情况，并且这两个人不是同一个人，通过比较两个人包含此limb的置信度来决定，
                        # 当前limb的节点应该分配给谁，同时把之前的那个与当前节点相连的节点(即partsA[i])从另一个人(subset)的节点集合中删除
                        if connectA[0] in subset[j1, :-2, 0]:
                            c1 = np.where(subset[j1, :-2, 0] == connectA[0])
                            c2 = np.where(subset[j2, :-2, 0] == connectA[1])
                        else:
                            c1 = np.where(subset[j1, :-2, 0] == connectA[1])
                            c2 = np.where(subset[j2, :-2, 0] == connectA[0])

                        # c1, c2分别是当前limb连接到j1人的第c1个关节点，j2人的第c2个关节点
                        c1 = int(c1[0])
                        c2 = int(c2[0])
                        assert c1 != c2, "an candidate keypoint is used twice, shared by two people"

                        # 如果当前考察的limb置信度比已经存在的两个人连接的置信度小，则跳过，否则删除已存在的不可信的连接节点。
                        if connectA[2] < subset[j1][c1][1] and connectA[2] < subset[j2][c2][1]:
                            continue  # the trick here is useful

                        small_j = j1
                        big_j = j2
                        remove_c = c1

                        if subset[j1][c1][1] > subset[j2][c2][1]:
                            small_j = j2
                            big_j = j1
                            remove_c = c2
                        # 删除和当前limb有连接,并且置信度低的那个人的节点
                        if params['remove_recon'] > 0:
                            subset[small_j][-2][0] -= candidate[subset[small_j][remove_c][0].astype(int), 2] + \
                                                      subset[small_j][remove_c][1]
                            subset[small_j][remove_c][0] = -1
                            subset[small_j][remove_c][1] = -1
                            subset[small_j][-1][0] -= 1

                # if find no partA in the subset, create a new subset
                # 如果肢体组成的关节点A,B没有被连接到某个人体则组成新的人体
                # ------------------------------------------------------------------
                #    1.Sort each possible connection by its score.
                #    2.The connection with the highest score is indeed a final connection.
                #    3.Move to next possible connection. If no parts of this connection have
                #    been assigned to a final connection before, this is a final connection.
                #    第三点是说，如果下一个可能的连接没有与之前的连接有共享端点的话，会被视为最终的连接，加入row
                #    4.Repeat the step 3 until we are done.
                # 说明见：　https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8
                # =================================================================
                elif not found and pair_id < len(limb_pairs):
                    # TOCHECK: Fixme: 检查一下是否正确
                    #  原始的时候是 k<18,因为我加了limb，所以是24,因为真正的limb是0~16，最后两个17,18是额外的不是limb
                    #  但是后面画limb的时候没有把鼻子和眼睛耳朵的连线画上，要改进
                    # `connection_all`: (#connect, 6=(A_peak_id, B_peak_id, dist_prior, A_part_id, B_part_id, limb_len))
                    connectA_peak_ids, connectA_prior, connectA_limblen = connectA[:2], connectA[2], connectA[-1]
                    row = -1 * np.ones((20, 2))  # > (20, (kp_id, dist_prior)) = `-1`
                    row[indexA][0] = partAs[person_idx]  # > (#k1_type,)
                    row[indexA][1] = connectA_prior  # > (#cand_pairs, 6) -> dist_prior
                    row[indexB][0] = partBs[person_idx]  # > (#k2_type,)
                    row[indexB][1] = connectA_prior  # > (#cand_pairs, 6) -> dist_prior
                    # > assign to 19 and 20
                    row[-1][0] = 2  # TOCHECK: why assign `2` here? is it `bg` class?
                    row[-1][1] = connectA_limblen  # `limb_len`: 这一位用来记录上轮连接limb时的长度，用来作为下一轮连接的先验知识
                    # > `candidate`: (#kp * person, (x,y,score,id)) -> peak_ids -> score
                    row[-2][0] = sum(candidate[connectA_peak_ids.astype(int), 2]) + connectA_prior  # > TOCHECK: `reverse_kp`?
                    # 两个端点的置信度+limb连接的置信度
                    # print('create a new subset:  ', row, '\t')
                    row = row[np.newaxis, :, :]  # 为了进行concatenate，需要插入一个轴, -> (1, 20, 2)
                    subset = np.concatenate((subset, row), axis=0)  # -> (1, 20, 2) TOCHECK: line 384?

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for person_idx in range(len(subset)):  # > #person
        # (params['thre1'] + params['thre2']) / 2:  # todo: tune, it matters much!
        if subset[person_idx][-1][0] < 4 or subset[person_idx][-2][0] / subset[person_idx][-1][0] < 0.45:
            deleteIdx.append(person_idx)
    subset = np.delete(subset, deleteIdx, axis=0)

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
        keypoints.append((person_keypoint_coordinates_coco, 1 - 1.0 / s[-2]))  # `s[19]` is the score

    for person_idx in range(len(keypoints)):  # > #person, `keypoints`: (#person, 2, #kp)
        print('the {}th keypoint detection result is : '.format(person_idx), keypoints[person_idx])

    canvas = cv2.imread(input_image)  # B,G,R order
    # canvas = oriImg

    # 画所有的峰值
    # for i in range(18):
    #     #     rgba = np.array(cmap(1 - i/18. - 1./36))
    #     #     rgba[0:3] *= 255
    #     for j in range(len(all_peaks[i])):  # all_peaks保存了坐标，score以及id
    #         # 注意x,y坐标谁在前谁在后，在这个project中有点混乱
    #         cv2.circle(canvas, all_peaks[i][j][0:2], 3, colors[i], thickness=-1)

    # 画所有的骨架
    color_board = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    color_idx = 0
    for limb_idx in draw_list:  # 画出18个limb　Fixme：我设计了25个limb,画的limb顺序需要调整，相应color数也要增加
        for person_idx in range(len(subset)):  # > `subset`: (#person, 20, 2) -> #person
            # `subset` -> person_idx -> limb_pairs -> pair_idx -> `candidate` -> pair location
            limb_pair = np.array(limb_pairs[limb_idx])  # > (kp_idx1, kp_idx2)
            index = subset[person_idx][limb_pair][..., 0]  # > (2, 2) -> (2,)
            if -1 in index:  # 有-1说明没有对应的关节点与之相连,即有一个类型的part没有缺失，无法连接成limb
                continue
            # 在上一个cell中有　canvas = cv2.imread(test_image) # B,G,R order
            cur_canvas = canvas.copy()
            # > `candidate`: (#kp * #person, (x,y,score,id))
            Y = candidate[index.astype(int), 0]  # > TOCHECK: (2,), isn't it `x` rather than `y`?
            X = candidate[index.astype(int), 1]  # > TOCHECK: (2,)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5  # TOCHECK: ??
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            # > SEE: bit.ly/2VvPaVt
            polygon = cv2.ellipse2Poly(center=(int(mY), int(mX)), axes=(int(length / 2), 3),
                                       angle=int(angle), arcStart=0, arcEnd=360, delta=1)

            cv2.circle(cur_canvas, center=(int(Y[0]), int(X[0])), radius=4, color=[0, 0, 0], thickness=2)
            cv2.circle(cur_canvas, center=(int(Y[1]), int(X[1])), radius=4, color=[0, 0, 0], thickness=2)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[color_board[color_idx]])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        color_idx += 1
    return canvas


if __name__ == '__main__':
    input_image = args.image
    output = args.output

    posenet = NetworkEval(opt, config, bn=True)
    print('> Model = ', posenet)
    print('Resuming from checkpoint ...... ')

    # #################################################
    # from collections import OrderedDict
    #
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint['weights'].items():
    #     # if 'out' in k or 'merge' in k:
    #     #     continue
    #     name = 'module.' + k  # add prefix 'module.'
    #     new_state_dict[name] = v
    # posenet.load_state_dict(new_state_dict)  # , strict=False
    # # #################################################

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

    tic = time.time()
    print('start processing...')
    # load config
    params, model_params = config_reader()
    tic = time.time()
    # generate image with body parts
    with torch.no_grad():
        canvas = process(input_image, params, model_params,
                         config.heat_layers + 2,  # `heat_layers`: 18 + 2(bg, reverse)
                         config.paf_layers)  # > `paf_layers`: 30

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    # TODO: the prediction is slow, how to fix it? Not solved yet. see:
    #  https://github.com/anatolix/keras_Realtime_Multi-Person_Pose_Estimation/issues/5

    # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)  # cv2.WINDOW_NORMAL 自动适合的窗口大小
    # cv2.imshow('result', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(output, canvas)

    # pdf = PdfPages(output + '.pdf')
    # plt.figure()
    # plt.plot(canvas[:, :, [2, 1, 0]])
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
    # pdf.savefig()
    # plt.close()
    # pdf.close()

    # dummy_input = torch.randn(1, 384, 384, 3)
    # from thop import profile
    # from thop import clever_format
    # flops, params = profile(posenet, inputs=(dummy_input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)
