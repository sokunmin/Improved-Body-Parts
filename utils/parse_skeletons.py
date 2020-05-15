
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
from utils import util
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

NUM_KEYPOINTS = 18
NUM_HEATMAPS = NUM_KEYPOINTS + 2
NUM_PAFS = 30


def predict_refactor(image, model, test_cfg, model_cfg, input_image_path, flip_avg=True, config=None):
    # > scale feature maps up to image size
    img_h, img_w, _ = image.shape

    heatmap_avg, paf_avg = None, None
    # > [1] scale search
    multiplier = [x * model_cfg['boxsize'] / img_h for x in test_cfg['scale_search']]
    # > [2] fix scale
    multiplier = [1.]  # > [0.5, 1., 1.5, 2., 3.]
    rotate_angle = test_cfg['rotation_search']  # > 0.0
    max_downsample = model_cfg['max_downsample']
    pad_value = model_cfg['padValue']
    stride = model_cfg['stride']
    flip_heat_ord = config.flip_heat_ord
    flip_paf_ord = config.flip_paf_ord

    for item in product(multiplier, rotate_angle):  # > #scales
        scale, angle = item
        img_max_h, img_max_w = (2600, 3800)  # CHANGED: (2300, 3200)->(2600,3800)
        if scale * img_h > img_max_h or scale * img_w > img_max_w:
            scale = min(img_max_h / img_h, img_max_w / img_w)
            print("Input image: '{}' is too big, shrink it!".format(input_image_path))

        # > `imageToTest`: (scaleH, scaleW, 3)
        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # > `imageToTest_padded`: (scale_padH, scale_padW, 3)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest,
                                                          max_downsample,  # > 64
                                                          pad_value)  # > 128
        scale_padH, scale_padW, _ = imageToTest_padded.shape

        # > WARN: `[1-1]`: we use OpenCV to read image`(BGR)` all the time
        input_img = np.float32(imageToTest_padded / 255)

        # > `[1-2]` :add rotate image
        if angle != 0:  # ADDED
            rotate_matrix = cv2.getRotationMatrix2D((input_img.shape[0] / 2, input_img.shape[1] / 2), angle, 1)
            rotate_matrix_reverse = cv2.getRotationMatrix2D((input_img.shape[0] / 2, input_img.shape[1] / 2), -angle, 1)
            input_img = cv2.warpAffine(input_img, rotate_matrix, (0, 0))

        # > `[1-2]` :add flip image
        swap_image = input_img[:, ::-1, :].copy()
        # plt.imshow(swap_image[:, :, [2, 1, 0]])  # Opencv image format: BGR
        # plt.show()
        input_img = np.concatenate((input_img[None, ...], swap_image[None, ...]), axis=0)  # (2, H, W, C)
        input_img = torch.from_numpy(input_img).cuda()

        # > `[1-3]-model`(4,)=(2, 50, featH, featW) x 4, `dtype=float16`
        output_tuple = model(input_img)  # > NOTE: feed img here -> (#stage, #scales, #img, 50, H, W)

        # > `[1-4]`: scales vary according to input image size.
        # > `-1`: last stage, `0`: high-res featmaps
        output = output_tuple[-1][0].cpu().numpy()  # -> (2, 50, featH, featW)

        output_blob = output[0].transpose((1, 2, 0))  # > (featH, featW, 50)
        output_paf = output_blob[:, :, :config.paf_layers]  # > `PAF`:(featH, featW, 30)
        output_heatmap = output_blob[:, :, config.paf_layers:config.num_layers]  # > `KP`:(featH, featW, 20)
        # > flipped image output
        output_blob_flip = output[1].transpose((1, 2, 0))  # > (featH, featW, 50)
        output_paf_flip = output_blob_flip[:, :, :config.paf_layers]  # `PAF`: (featH, featW, 30)
        output_heatmap_flip = output_blob_flip[:, :, config.paf_layers:config.num_layers]  # > `KP`: (featH, featW, 20)

        # > `[1-5]`: flip ensemble & average
        if flip_avg:
            output_paf_avg = (output_paf + output_paf_flip[:, ::-1, :][:, :, flip_paf_ord]) / 2  # > (featH, featW, 30)
            output_heatmap_avg = (output_heatmap + output_heatmap_flip[:, ::-1, :][:, :,
                                                   flip_heat_ord]) / 2  # > (featH, featW, 20)
        else:
            output_paf_avg = output_paf  # > (featH, featW, 30)
            output_heatmap_avg = output_heatmap  # > (featH, featW, 20)

        if angle != 0:  # ADDED
            output_heatmap_avg = cv2.warpAffine(output_heatmap_avg, rotate_matrix_reverse, (0, 0))
            output_paf_avg = cv2.warpAffine(output_paf_avg, rotate_matrix_reverse, (0, 0))
        heatmap_avg = output_heatmap_avg
        paf_avg = output_paf_avg
    return heatmap_avg, paf_avg


def find_peaks_refactor(param, img):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """

    peaks_binary = (
       maximum_filter(img, footprint=generate_binary_structure(2, 1)) == img) * (img > param)  # > (H, W)
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y x]...]

    return np.array(np.nonzero(peaks_binary)[::-1]).T  # > (2,1) -> (1,2)


def compute_resized_coords(coords, resizeFactor):
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5


def heatmap_nms(heatmaps, upsample_factor=1., bool_refine_center=True):

    joint_list_per_joint_type = []
    cnt_total_joints = 0

    # For every peak found, `win_size` specifies how many pixels in each
    # direction from the peak we take to obtain the patch that will be
    # upsampled. Eg: `win_size`=1 -> patch is 3x3; `win_size`=2 -> 5x5
    # (for BICUBIC interpolation to be accurate, `win_size` needs to be >=2!)
    win_size = 2

    for joint in range(NUM_KEYPOINTS):
        map_orig = heatmaps[:, :, joint]  # > (featH, featW)
        peak_coords = find_peaks_refactor(0.1, map_orig)  # `thres`: 0.1
        peaks = np.zeros((len(peak_coords), 4))  # > (#peaks, (x,y,score,peak_id))
        for i, peak in enumerate(peak_coords):
            if bool_refine_center:
                x_min, y_min = np.maximum(0, peak - win_size)
                x_max, y_max = np.minimum(np.array(map_orig.T.shape) - 1, peak + win_size)

                # Take a small patch around each peak and only upsample that
                # tiny region
                patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]  # (5, 5)
                map_upsample = cv2.resize(patch,
                                          None,
                                          fx=upsample_factor,
                                          fy=upsample_factor,
                                          interpolation=cv2.INTER_CUBIC)  # (40, 40)

                # Obtain the coordinates of the maximum value in the patch
                location_of_max = np.unravel_index(map_upsample.argmax(), map_upsample.shape)
                # Remember that peaks indicates [x,y] -> need to reverse it for [y,x]
                # `upsample_factor`: 8, this maps to feat coord
                location_of_patch_center = compute_resized_coords(peak[::-1] - [y_min, x_min], upsample_factor)
                # Calculate the offset wrt to the patch center where the actual maximum is

                refined_center = (location_of_max - location_of_patch_center)
                peak_score = map_upsample[location_of_max]
            else:
                refined_center = [0, 0]
                # Flip peak coordinates since they are [x,y] instead of [y,x]
                peak_score = map_orig[tuple(peak[::-1])]

            peaks[i, :] = tuple(
                x for x in compute_resized_coords(peak_coords[i], upsample_factor) +
                refined_center[::-1]
            ) + (peak_score, cnt_total_joints)
            cnt_total_joints += 1
        joint_list_per_joint_type.append(peaks)

    return joint_list_per_joint_type  # > (#kp_types, (#peaks, (x,y,score,peak_id)))


# =======================================================================================
def predict(image, model, test_cfg, model_cfg, input_image_path, flip_avg=True, config=None):
    # > scale feature maps up to image size
    img_h, img_w, _ = image.shape
    heatmap_avg = np.zeros((img_h, img_w, NUM_HEATMAPS))  # > `heatmap_avg`: (imgH, imgW, 20)
    paf_avg = np.zeros((img_h, img_w, NUM_PAFS))  # > `paf_layers`: (imgH, imgW, 30)
    # > [1] scale search
    multiplier = [x * model_cfg['boxsize'] / img_h for x in test_cfg['scale_search']]
    # > [2] fix scale
    multiplier = [1.]  # > [0.5, 1., 1.5, 2., 3.]
    rotate_angle = test_cfg['rotation_search']  # > 0.0
    max_downsample = model_cfg['max_downsample']
    pad_value = model_cfg['padValue']
    stride = model_cfg['stride']
    flip_heat_ord = config.flip_heat_ord
    flip_paf_ord = config.flip_paf_ord

    for item in product(multiplier, rotate_angle):  # > #scales
        scale, angle = item
        img_max_h, img_max_w = (2600, 3800)  # CHANGED: (2300, 3200)->(2600,3800)
        if scale * img_h > img_max_h or scale * img_w > img_max_w:
            scale = min(img_max_h / img_h, img_max_w / img_w)
            print("Input image: '{}' is too big, shrink it!".format(input_image_path))

        # > `imageToTest`: (scaleH, scaleW, 3)
        imageToTest = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # > `imageToTest_padded`: (scale_padH, scale_padW, 3)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest,
                                                          max_downsample,  # > 64
                                                          pad_value)  # > 128
        scale_padH, scale_padW, _ = imageToTest_padded.shape

        # > WARN: `[1-1]`: we use OpenCV to read image`(BGR)` all the time
        input_img = np.float32(imageToTest_padded / 255)

        # > `[1-2]` :add rotate image
        if angle != 0:  # ADDED
            rotate_matrix = cv2.getRotationMatrix2D((input_img.shape[0] / 2, input_img.shape[1] / 2), angle, 1)
            rotate_matrix_reverse = cv2.getRotationMatrix2D((input_img.shape[0] / 2, input_img.shape[1] / 2), -angle, 1)
            input_img = cv2.warpAffine(input_img, rotate_matrix, (0, 0))

        # > `[1-2]` :add flip image
        swap_image = input_img[:, ::-1, :].copy()
        # plt.imshow(swap_image[:, :, [2, 1, 0]])  # Opencv image format: BGR
        # plt.show()
        input_img = np.concatenate((input_img[None, ...], swap_image[None, ...]), axis=0)  # (2, H, W, C)
        input_img = torch.from_numpy(input_img).cuda()

        # > `[1-3]-model`(4,)=(2, 50, featH, featW) x 4, `dtype=float16`
        output_tuple = model(input_img)  # > NOTE: feed img here -> (#stage, #scales, #img, 50, H, W)

        # > `[1-4]`: scales vary according to input image size.
        # > `-1`: last stage, `0`: high-res featmaps
        output = output_tuple[-1][0].cpu().numpy()  # -> (2, 50, featH, featW)

        output_blob = output[0].transpose((1, 2, 0))  # > (featH, featW, 50)
        output_paf = output_blob[:, :, :config.paf_layers]  # > `PAF`:(featH, featW, 30)
        output_heatmap = output_blob[:, :, config.paf_layers:config.num_layers]  # > `KP`:(featH, featW, 20)
        # > flipped image output
        output_blob_flip = output[1].transpose((1, 2, 0))  # > (featH, featW, 50)
        output_paf_flip = output_blob_flip[:, :, :config.paf_layers]  # `PAF`: (featH, featW, 30)
        output_heatmap_flip = output_blob_flip[:, :, config.paf_layers:config.num_layers]  # > `KP`: (featH, featW, 20)

        # > `[1-5]`: flip ensemble & average
        if flip_avg:
            output_paf_avg = (output_paf + output_paf_flip[:, ::-1, :][:, :, flip_paf_ord]) / 2  # > (featH, featW, 30)
            output_heatmap_avg = (output_heatmap + output_heatmap_flip[:, ::-1, :][:, :, flip_heat_ord]) / 2  # > (featH, featW, 20)
        else:
            output_paf_avg = output_paf  # > (featH, featW, 30)
            output_heatmap_avg = output_heatmap  # > (featH, featW, 20)

        # > `[1-6]`: extract outputs, resize, and remove padding
        # > `heatmap`: (featH, featW, 20) -> (scale_padH, scale_padW, 20)
        heatmap = cv2.resize(output_heatmap_avg,  # > `KP`: (featH, featW, 20)
                             (0, 0),
                             fx=stride,  # > `stride`: 4
                             fy=stride,
                             interpolation=cv2.INTER_CUBIC)

        # > `paf`: (featH, featW, 30) -> (scale_padH, scale_padW, 30)
        paf = cv2.resize(output_paf_avg,
                         (0, 0),
                         fx=stride,  # > `stride`: 4
                         fy=stride,
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

    return heatmap_avg, paf_avg


def find_peaks(heatmap_avg, test_cfg):
    all_peaks = []
    peak_counter = 0

    # > `heatmap_avg`: (imgH, imgW, 20)
    heatmap_avg = heatmap_avg.astype(np.float32)

    # > (imgH, imgW, 20) -> (imgH, imgW, 18) -> (18, imgH, imgW) -> (1, 18, imgH, imgW)
    filter_map = heatmap_avg[:, :, :NUM_KEYPOINTS].copy().transpose((2, 0, 1))[None, ...]
    filter_map = torch.from_numpy(filter_map).cuda()

    # > (1, 18, imgH, imgW), `thre1`: 0.1
    filter_map = util.keypoint_heatmap_nms(filter_map, kernel=3, thre=test_cfg['thre1'])
    filter_map = filter_map.cpu().numpy().squeeze().transpose((1, 2, 0))  # > (imgH, imgW, 18)
    # > `heatmap_avg`: (imgH, imgW, 20)
    for part in range(NUM_KEYPOINTS):  # > `#kp`: 沒有對背景（序號19）取非極大值抑制NMS
        map_orig = heatmap_avg[:, :, part]  # > (imgH, imgW)
        # NOTE: 在某些情况下，需要对一个像素的周围的像素给予更多的重视。因此，可通过分配权重来重新计算这些周围点的值。
        # 这可通过高斯函数（钟形函数，即喇叭形数）的权重方案来解决。
        peaks_binary = filter_map[:, :, part]  # > (imgH, imgW)
        peak_y, peak_x = np.nonzero(peaks_binary)
        peaks = list(zip(peak_x, peak_y))  # > (#peaks, (x,y))
        # > `offset_radius`: 2, `refined_peaks_with_score`: (x,y,score)
        refined_peaks_with_score = [util.refine_centroid(map_orig, anchor, test_cfg['offset_radius']) for anchor in peaks]

        # > `id`: [0, #peaks), `refined_peaks_with_score`: (#peaks, (x,y,score))
        id = range(peak_counter, peak_counter + len(refined_peaks_with_score))  # `id`: len(x) = #peaks

        # > [(x,y,score) + (id,) = (x,y,score,id)] of `certain type` of keypoint.
        # 为每一个相应peak (parts)都依次编了一个号
        peaks_with_score_and_id = [refined_peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)  # refined_peaks

    return all_peaks  # > (#kp_types, (x,y,score,kp_id))


def find_connections(all_peaks, paf_avg, img_height, test_cfg, joint2limb_pairs):
    # > `all_peaks`: (#kp, (x,y,score,kp_id)
    connected_limbs = []  # > (#connect, 6=(src_peak_id, dst_peak_id, score, joint_src_id, joint_dst_id, limb_len))
    special_limb = []
    paf_thresh = test_cfg['thre2']
    connect_ration = test_cfg['connect_ration']
    # > `#limb` = `#connection` = `#paf_channel`, `limb_pairs[i]`=(kp_id1, kp_id2)
    for pair_id in range(len(joint2limb_pairs)):  # > 30 pairs, 最外层的循环是某一个limb_pairs，因为mapIdx个数与之是一致对应的
        # 某一个channel上limb的响应热图, 它的长宽与原始输入图片大小一致，前面经过resize了
        score_mid = paf_avg[:, :, pair_id]  # (imgH, imgW, 30) -> `c=k` -> (imgH, imgW)

        # `all_peaks(list)`: (#kp, #peaks, (x,y,score,id)), 每一行也是一个list,保存了检测到的特定的parts(joints)
        joints_src = all_peaks[joint2limb_pairs[pair_id][0]]  # > `all_peaks` -> `kp_id1` -> (x,y,score,id)
        joints_dst = all_peaks[joint2limb_pairs[pair_id][1]]  # > `all_peaks` -> `kp_id2` -> (x,y,score,id)

        if len(joints_src) == 0 and len(joints_dst) == 0:
            special_limb.append(pair_id)
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

                    mid_num = min(int(round(limb_len + 1)), test_cfg['mid_num'])  # > `mid_num`: 20

                    # TOCHECK: failure case when 2 body parts overlaps
                    if limb_len == 0:  # 为了跳过出现不同节点相互覆盖出现在同一个位置，也有说norm加一个接近0的项避免分母为0
                        # SEE：https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/54
                        continue

                    # > [new]: (20,) , MY-TODO: re-written into C++
                    limb_intermed_x = np.round(np.linspace(start=joint_src[0], stop=joint_dst[0], num=mid_num)).astype(np.intp)
                    limb_intermed_y = np.round(np.linspace(start=joint_src[1], stop=joint_dst[1], num=mid_num)).astype(np.intp)
                    limb_response = score_mid[limb_intermed_y, limb_intermed_x]  # > (20,)

                    score_midpts = limb_response  # > (mid_num,)
                    # > `score_with_dist_prior`: scalar
                    connect_score = score_midpts.mean() + min(0.5 * img_height / limb_len - 1, 0)
                    # 这一项是为了惩罚过长的connection, 只有当长度大于图像高度的一半时才会惩罚 todo
                    # The term of sum(score_midpts)/len(score_midpts), see the link below.
                    # https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/issues/48
                    # > `thre2`: 0.1, `connect_ration`: 0.8 -> `criterion1`: True/False
                    criterion1 = \
                        np.count_nonzero(score_midpts > paf_thresh) >= mid_num * connect_ration
                    # 我认为这个判别标准是保证paf朝向的一致性  threshold = param['thre2'] =0.12
                    # CMU原始项目中parm['thre2'] = 0.05
                    criterion2 = connect_score > 0  # > True/False

                    if criterion1 and criterion2:
                        # > TOCHECK: [0.5, 0.25, 0.25] -> (candA_id, candB_id, dist_prior, limb_len, `confidence`)
                        connection_candidates.append([
                            i, j,
                            connect_score,
                            limb_len,
                            # TOCHECK: weighted sum?
                            #  CHANGED: can be optimized if weights are removed
                            0.5 * connect_score +
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
            for potential_connect in connection_candidates:  # 根據confidence的順序選擇connections
                i, j, s, limb_len = potential_connect[0:4]
                if i not in connections[:, 3] and j not in connections[:, 4]:
                    # 进行判断确保不会出现两个端点集合A,B中，出现一个集合中的点与另外一个集合中两个点同时相连
                    connections = np.vstack([
                        connections,
                        # `src_peak_id`, `dst_peak_id`, `dist_prior`, `joint_src_id`, `joint_dst_id`, `limb_len`
                        [joints_src[i][3], joints_dst[j][3], s, i, j, limb_len]
                    ])  # > (#connect, 6)

                    if len(connections) >= max_connections:  # 會出現關節點不夠連的情況
                        break
            connected_limbs.append(connections)

    return connected_limbs, special_limb


def find_humans(connected_limbs, special_limb, joint_list, test_cfg, joint2limb_pairs):
    """
     `connected_limbs`: (#connect, 6=(A_peak_id, B_peak_id, dist_prior, A_part_id, B_part_id, limb_len))
     `connected_limbs[k]`: 保存的是第k个类型的所有limb连接，可能有多个，也可能一个没有
     `connected_limbs` 每一行是一个类型的limb, 每一行格式: N * [idA, idB, score, i, j]
      `skeletons`: 每一行对应的是 (一个人, 18个关键点, (number, score的结果)) = (#person, 18, 2) + (#person, 2, 2)
     `connected_limbs` 每一行的list保存的是一类limb(connection),遍历所有此类limb,一般的有多少个特定的limb就有多少个人
        # `special_K` ,表示没有找到关节点对匹配的肢体

    """
    # last number in each row is the `total parts number of that person`
    # the second last number in each row is `the score of the overall configuration`
    person_to_joint_assoc = -1 * np.ones((0, NUM_KEYPOINTS + 2, 2))  # > (#person, 20, 2)
    # `joint_list` -> `candidate`: (#kp * person, (x,y,score,peak_id))
    joint_candidates = np.array([item for sublist in joint_list for item in sublist])

    len_rate = test_cfg['len_rate']
    connection_tole = test_cfg['connection_tole']
    delete_shared_joints = test_cfg['remove_recon']

    for limb_type in range(len(joint2limb_pairs)):  # > #pairs
        if limb_type not in special_limb:  # > limb is connected and exists (PAF)
            joint_src_type, joint_dst_type = joint2limb_pairs[limb_type]

            for limb_id, limb_info in enumerate(connected_limbs[limb_type]):
                limb_src_peak_id = limb_info[0]
                limb_dst_peak_id = limb_info[1]
                limb_connect_score = limb_info[2]
                limb_len = limb_info[-1]

                person_assoc_idx = []
                for person_id, person1_limbs in enumerate(person_to_joint_assoc):  # > #person
                    if person1_limbs[joint_src_type, 0] == limb_src_peak_id or \
                       person1_limbs[joint_dst_type, 0] == limb_dst_peak_id:
                        # check if two joints of a limb is used in previous step, which means it is used by someone.
                        if len(person_assoc_idx) >= 2:
                            print('************ error occurs! 3 joints sharing have been found  *******************')
                            continue
                        person_assoc_idx.append(person_id)

                if len(person_assoc_idx) == 1:
                    person1_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                    person1_dst_peak_id = person1_limbs[joint_dst_type, 0]
                    person1_dst_connect_score = person1_limbs[joint_dst_type, 1]
                    person1_limb_len = person1_limbs[-1, 1]
                    # > `len_rate`:16
                    if person1_dst_peak_id.astype(int) == -1 and person1_limb_len * len_rate > limb_len:
                        # > (1) the length of new limb is longer than existed one, discard it.
                        # > (2) no joints for current person, assign current joint to this guy.
                        # 这一个判断非常重要，因为第18和19个limb分别是 2->16, 5->17,这几个点已经在之前的limb中检测到了，
                        # 所以如果两次结果一致，不更改此时的part分配，否则又分配了一次，编号是覆盖了，但是继续运行下面代码，part数目
                        # 会加１，结果造成一个人的part之和>18。不过如果两侧预测limb端点结果不同，还是会出现number of part>18，造成多检
                        # TOCHECK: 没有利用好冗余的connection信息，最后两个limb的端点与之前循环过程中重复了，但没有利用聚合，
                        #  只是直接覆盖，其实直接覆盖是为了弥补漏检

                        person1_limbs[joint_dst_type] = [limb_dst_peak_id, limb_connect_score]
                        person1_limbs[-1, 0] += 1  # last number in each row is the total parts number of that person
                        person1_limbs[-1, 1] = max(limb_len, person1_limb_len)
                        person1_limbs[-2, 0] += joint_candidates[limb_dst_peak_id.astype(int), 2] + limb_connect_score
                        # the second last number in each row is the score of the overall configuration

                    # dst joint is connected, but `peak_id` is not same.
                    elif person1_dst_peak_id.astype(int) != limb_dst_peak_id.astype(int) and \
                         person1_dst_connect_score <= limb_connect_score and \
                         person1_limb_len * len_rate > limb_len:
                        # keep original id and score, and then replace lower score with higher one.
                        person1_limbs[joint_dst_type] = [limb_dst_peak_id, limb_connect_score]
                        person1_limbs[-2, 0] -= joint_candidates[person1_dst_peak_id.astype(int), 2] + person1_dst_connect_score
                        person1_limbs[-2, 0] += joint_candidates[limb_dst_peak_id.astype(int), 2] + limb_connect_score
                        # choose longer limb
                        person1_limbs[-1, 1] = max(limb_len, person1_limb_len)

                    # dst joint is connected, and peak_id is same, but prev score is lower than current one.
                    elif person1_dst_peak_id.astype(int) == limb_dst_peak_id.astype(int) and \
                         person1_dst_connect_score <= limb_connect_score:

                        person1_limbs[joint_dst_type] = [limb_dst_peak_id, limb_connect_score]
                        person1_limbs[-2, 0] -= joint_candidates[person1_dst_peak_id.astype(int), 2] + person1_dst_connect_score
                        person1_limbs[-2, 0] += joint_candidates[limb_dst_peak_id.astype(int), 2] + limb_connect_score
                        person1_limbs[-1, 1] = max(limb_len, person1_limb_len)

                elif len(person_assoc_idx) == 2:  # if found 2 and disjoint, merge them (disjoint：不相交)
                    # If humans `H1` and `H2` share a part index with the same coordinates,
                    #  they are sharing the same part!
                    #  `H1` and `H2` are, therefore, the same humans.
                    #   So we merge both sets into `H1` and remove `H2`.
                    # SEE: https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8

                    person1_id, person2_id = person_assoc_idx[0], person_assoc_idx[1]
                    person1_limbs = person_to_joint_assoc[person1_id]  # > (20,2)
                    person2_limbs = person_to_joint_assoc[person2_id]  # > (20,2)
                    person1_limb_len = person1_limbs[-1, 1]

                    membership1 = ((person1_limbs[..., 0] >= 0).astype(int))[:-2]  # > (18,)
                    membership2 = ((person2_limbs[..., 0] >= 0).astype(int))[:-2]  # > (18,)
                    membership = membership1 + membership2  # > (18,)
                    # > this joint corresponds to the same person, increase joint count of this person.
                    if len(np.nonzero(membership == 2)[0]) == 0:  # if found 2 and disjoint, merge them

                        min_limb1 = np.min(person1_limbs[:-2, 1][membership1 == 1])  # (18,) -> (#pos,) -> scalar
                        min_limb2 = np.min(person2_limbs[:-2, 1][membership2 == 1])  # (18,) -> (#pos,) -> scalar
                        min_tolerance = min(min_limb1, min_limb2)  # > min confidence
                        if limb_connect_score >= connection_tole * min_tolerance or \
                           person1_limb_len * len_rate > limb_len:
                            # > do not connect if confidence is low or current limb is longer than existed one.
                            # continue
                            person1_limbs[:-2] += (person2_limbs[:-2] + 1)  # > (18,2)

                            # 对于没有节点标记的地方，因为两行subset相应位置处都是-1,所以合并之后没有节点的部分依旧是-１
                            # 把不相交的两个subset[j1],[j2]中的id号进行相加，从而完成合并，这里+1是因为默认没有找到关键点初始值是-1
                            person1_limbs[-2:][:, 0] += person2_limbs[-2:][:, 0]  # 两行subset的点的个数和总置信度相加

                            person1_limbs[-2, 0] += limb_connect_score
                            person1_limbs[-1, 1] = max(limb_len, person1_limb_len)
                            # 注意：　因为是disjoint的两行person_to_joint_assoc点的merge，因此先前存在的节点的置信度之前已经被加过了 !! 这里只需要再加当前考察的limb的置信度
                            person_to_joint_assoc = np.delete(person_to_joint_assoc, person2_id, 0)

                    else:
                        # this joint is shared by two different persons.
                        # We assign joint to who has higher confidence of a limb and
                        # delete the connected joint from another guy.
                        if delete_shared_joints > 0:
                            person1_peak_ids = person1_limbs[:-2, 0]  # > (18, 2) -> (18,)
                            person2_peak_ids = person2_limbs[:-2, 0]  # > (18, 2) -> (18,)
                            if limb_src_peak_id in person1_peak_ids:
                                conn1_idx = int(np.where(person1_peak_ids == limb_src_peak_id)[0])
                                conn2_idx = int(np.where(person2_peak_ids == limb_dst_peak_id)[0])
                            else:
                                conn1_idx = int(np.where(person1_peak_ids == limb_dst_peak_id)[0])
                                conn2_idx = int(np.where(person2_peak_ids == limb_src_peak_id)[0])

                            # c1, c2分别是当前limb连接到j1人的第c1个关节点，j2人的第c2个关节点
                            assert conn1_idx != conn2_idx, "an candidate keypoint is used twice, shared by two people"
                            # > skip this joint if confidence of current limb is lower than the one connected by two guys
                            if limb_connect_score >= person1_limbs[conn1_idx, 1] and \
                               limb_connect_score >= person2_limbs[conn2_idx, 1]:
                                # continue  # the trick here is useful

                                # > delete less confident joint
                                if person1_limbs[conn1_idx, 1] > person2_limbs[conn2_idx, 1]:
                                    low_conf_idx = person2_id
                                    high_conf_idx = person1_id
                                    delete_conn_idx = conn2_idx
                                else:
                                    low_conf_idx = person1_id
                                    high_conf_idx = person2_id
                                    delete_conn_idx = conn1_idx
                                # > delete the joint that has low confidence connected with current limb
                                # > TOCHECK: detect more joints if not delete?
                                if delete_shared_joints > 0:
                                    # `person_to_joint_assoc`: (#person, 20, 2)
                                    person_to_joint_assoc[low_conf_idx, -2, 0] -= \
                                        joint_candidates[person_to_joint_assoc[low_conf_idx, delete_conn_idx, 0].astype(int), 2] + \
                                        person_to_joint_assoc[low_conf_idx, delete_conn_idx, 1]
                                    person_to_joint_assoc[low_conf_idx, delete_conn_idx, 0] = -1
                                    person_to_joint_assoc[low_conf_idx, delete_conn_idx, 1] = -1
                                    person_to_joint_assoc[low_conf_idx, -1, 0] -= 1

                # No person has claimed any of these joints, create a new person
                # ------------------------------------------------------------------
                #    1.Sort each possible connection by its score.
                #    2.The connection with the highest score is indeed a final connection.
                #    3.Move to next possible connection. If no parts of this connection have
                #    been assigned to a final connection before, this is a final connection.
                #    第三点是说，如果下一个可能的连接没有与之前的连接有共享端点的话，会被视为最终的连接，加入row
                #    4.Repeat the step 3 until we are done.
                # SEE：　https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8

                elif limb_type < len(joint2limb_pairs):
                    row = -1 * np.ones((NUM_KEYPOINTS + 2, 2))  # > (20, (limb_type, dist_prior)) = `-1`

                    joint_peak_ids = limb_info[:2].astype(int)
                    # Store the joint info of the new connection
                    row[joint_src_type] = [limb_src_peak_id, limb_connect_score]  # > src_type
                    row[joint_dst_type] = [limb_dst_peak_id, limb_connect_score]  # > dst_type
                    # Total count of connected joints for this person: 2
                    row[-1] = [2, limb_len]  # > `limb_len`: 这一位用来记录上轮连接limb时的长度，用来作为下一轮连接的先验知识
                    # Compute overall score: score joint_src + score joint_dst + score connection
                    row[-2][0] = sum(joint_candidates[joint_peak_ids, 2]) + limb_connect_score
                    # > joint score + limb score
                    row = row[np.newaxis, :, :]
                    person_to_joint_assoc = np.concatenate((person_to_joint_assoc, row), axis=0)  # -> (1, 20, 2)
    # 将没有被分配到一些人身上的点分配给距离它们近，并且缺少此类节点的人身上？或许这样做坏处更多
    # Delete people who have very few parts connected
    people_to_delete = []
    for person_id, person_info in enumerate(person_to_joint_assoc):  # > #person
        if person_info[-1, 0] < 2 or person_info[-2, 0] / person_info[-1, 0] < 0.45:
            people_to_delete.append(person_id)
    person_to_joint_assoc = np.delete(person_to_joint_assoc, people_to_delete, axis=0)

    return person_to_joint_assoc, joint_candidates