#include <iostream>
#include <algorithm>
#include <math.h>
#include <assert.h>
#include "pafprocess.h"

#define PEAKS(i, j, k) peaks[k+p3*(j+p2*i)]
#define PAF(i, j, k) pafmap[k+f3*(j+f2*i)]

#define LIMB_INFO_IDX NUM_PART + 1
#define SCORE_IDX NUM_PART

using namespace std;

vector <vector<ConnectedPeak> > skeletons;  // (#skeleton, 20, 2)
vector <Peak> peak_infos_line;

int round2int(float v);

vector <float>
get_paf_scores(float *pafmap, int pair_id, int num_steps, int &f2, int &f3, Peak &peak1, Peak &peak2);

bool comp_candidate(ConnectionCandidate a, ConnectionCandidate b);

int process_paf(int p1, int p2, int p3, float *peaks, int h1, int h2, int h3, float *heatmap, int f1, int f2, int f3,
                float *pafmap) {
    // `joint_list`: (p1=#person, p2=18, p3=4)
    // `heatmap_upsamp`: (h1=H, h2=W, h3=20)
    // `paf_upsamp`: (f1=H, f2=W, f3=30)
    vector <Peak> peak_infos[NUM_PART];  // Peak: (x,y,score,id)
    int peak_cnt = 0;
    for (int img_id = 0; img_id < p1; img_id++){  // p1=#person
        for (int peak_index = 0; peak_index < p2; peak_index++) {  // p2=#kp
            Peak info;
            info.id = peak_cnt++;
            info.x = PEAKS(img_id, peak_index, 0);  // 0 + p3 * (j + p2 * i)
            info.y = PEAKS(img_id, peak_index, 1);  // 1 + p3 * (j + p2 * i)
            info.score = PEAKS(img_id, peak_index, 2);  // 2 + p3 * (j + p2 * i)
            int part_id = PEAKS(img_id, peak_index, 4);  // 4 + p3 * (j + p2 * i)
            peak_infos[part_id].push_back(info);
        }
    }

    peak_infos_line.clear();
    for (int part_id = 0; part_id < NUM_PART; part_id++) {
        for (int i = 0; i < (int) peak_infos[part_id].size(); i++) {
            peak_infos_line.push_back(peak_infos[part_id][i]);
        }
    }

    // Start to Connect
    vector <Connection> connection_all[COCOPAIRS_SIZE];
    for (int pair_id = 0; pair_id < COCOPAIRS_SIZE; pair_id++) {
        vector <ConnectionCandidate> candidates;
        vector <Peak> &peak_a_list = peak_infos[COCOPAIRS[pair_id][0]];
        vector <Peak> &peak_b_list = peak_infos[COCOPAIRS[pair_id][1]];

        if (peak_a_list.size() == 0 || peak_b_list.size() == 0) {
            continue;
        }

        for (int peak_a_id = 0; peak_a_id < (int) peak_a_list.size(); peak_a_id++) {
            Peak &peak_a = peak_a_list[peak_a_id];
            for (int peak_b_id = 0; peak_b_id < (int) peak_b_list.size(); peak_b_id++) {
                Peak &peak_b = peak_b_list[peak_b_id];

                // calculate vector(direction)
                VectorXY vec;
                vec.x = peak_b.x - peak_a.x;
                vec.y = peak_b.y - peak_a.y;
                float norm = (float) sqrt(vec.x * vec.x + vec.y * vec.y);
                if (norm < 1e-12) continue;
                vec.x = vec.x / norm;
                vec.y = vec.y / norm;

                int num_steps = min(round2int(norm), STEP_PAF);
                vector <float> paf_scores = get_paf_scores(
                        pafmap, pair_id, num_steps,
                        f2, f3,
                        peak_a, peak_b
                );
                float scores = 0.0f;

                // criterion 1 : score threshold count
                int criterion1 = 0;
                for (int i = 0; i < num_steps; i++) {
                    float score = paf_scores[i];
                    scores += score;

                    if (score > THRESH_PAF_SCORE) criterion1 += 1;
                }

                float criterion2 = scores / STEP_PAF + min(0.0, 0.5 * h1 / norm - 1.0);
                float min_num_steps = round2int(num_steps * THRESH_PAF_STEP_RATIO);
                if (criterion1 > min_num_steps && criterion2 > 0) {
                    ConnectionCandidate candidate;
                    candidate.idx1 = peak_a_id;
                    candidate.idx2 = peak_b_id;
                    candidate.score = criterion2;
                    candidate.etc = criterion2 + peak_a.score + peak_b.score;
                    candidate.length = norm;
                    candidates.push_back(candidate);
                }
            }
        }

        vector <Connection> &conns = connection_all[pair_id];
        sort(candidates.begin(), candidates.end(), comp_candidate);
        for (int c_id = 0; c_id < (int) candidates.size(); c_id++) {
            ConnectionCandidate &candidate = candidates[c_id];
            bool assigned = false;
            for (int conn_id = 0; conn_id < (int) conns.size(); conn_id++) {
                if (conns[conn_id].peak_id1 == candidate.idx1) {
                    // already assigned
                    assigned = true;
                    break;
                }
                if (assigned) break;
                if (conns[conn_id].peak_id2 == candidate.idx2) {
                    // already assigned
                    assigned = true;
                    break;
                }
                if (assigned) break;
            }
            if (assigned) continue;

            Connection conn;
            conn.peak_id1 = peak_a_list[candidate.idx1].id;
            conn.peak_id2 = peak_b_list[candidate.idx2].id;
            conn.score = candidate.score;
            conn.cid1 = candidate.idx1;
            conn.cid2 = candidate.idx2;
            conn.length = candidate.length;
            conns.push_back(conn);
        }
    }

    // Generate skeletons
    skeletons.clear();
    for (int pair_id = 0; pair_id < COCOPAIRS_SIZE; pair_id++) {  // #limbs
        vector <Connection> &conns = connection_all[pair_id];
        int part_id1 = COCOPAIRS[pair_id][0];
        int part_id2 = COCOPAIRS[pair_id][1];

        for (int conn_id = 0; conn_id < (int) conns.size(); conn_id++) {  // #connect
            Connection *cur_conn = &conns[conn_id];
            int num_found = 0;
            int skeleton_idx1 = 0, skeleton_idx2 = 0;
            for (int skeleton_id = 0; skeleton_id < (int) skeletons.size(); skeleton_id++) {  // #person
                if (skeletons[skeleton_id][part_id1].id == cur_conn->peak_id1 ||
                    skeletons[skeleton_id][part_id2].id == cur_conn->peak_id2) {
                    if (num_found == 0) skeleton_idx1 = skeleton_id;
                    if (num_found == 1) skeleton_idx2 = skeleton_id;
                    num_found += 1;
                }
            }

            if (num_found == 1) {
                vector<ConnectedPeak> *skeleton1 = &skeletons[skeleton_idx1];
                int skeleton1_limb_len = (*skeleton1)[LIMB_INFO_IDX].length * LIMB_LENGTH_RATE;
                // if other joint is not associated to anyone yet.
                if ((*skeleton1)[part_id2].id == -1 && skeleton1_limb_len > cur_conn->length) {
                    // associate it with current skeleton
                    (*skeleton1)[part_id2].id = cur_conn->peak_id2;
                    (*skeleton1)[part_id2].score = cur_conn->score;
                    // increase number of limbs associated to this skeleton and update length
                    (*skeleton1)[LIMB_INFO_IDX].count += 1;
                    (*skeleton1)[LIMB_INFO_IDX].length = max((*skeleton1)[LIMB_INFO_IDX].length, cur_conn->length);
                    // update total score
                    (*skeleton1)[SCORE_IDX].score += peak_infos_line[cur_conn->peak_id2].score + cur_conn->score;
                } else if ((*skeleton1)[part_id2].id != cur_conn->peak_id2 &&
                           (*skeleton1)[part_id2].score <= cur_conn->score &&
                           skeleton1_limb_len > cur_conn->length) {
                    (*skeleton1)[part_id2].id = cur_conn->peak_id2;
                    (*skeleton1)[part_id2].score = cur_conn->score;
                    (*skeleton1)[SCORE_IDX].score -= peak_infos_line[part_id2].score + (*skeleton1)[part_id2].score;
                    (*skeleton1)[SCORE_IDX].score += peak_infos_line[cur_conn->peak_id2].score + cur_conn->score;
                    (*skeleton1)[LIMB_INFO_IDX].length = max((*skeleton1)[LIMB_INFO_IDX].length, cur_conn->length);
                } else if ((*skeleton1)[part_id2].id == cur_conn->peak_id2 &&
                           (*skeleton1)[part_id2].score <= cur_conn->score) {
                    (*skeleton1)[part_id2].id = cur_conn->peak_id2;
                    (*skeleton1)[part_id2].score = cur_conn->score;
                    (*skeleton1)[SCORE_IDX].score -= peak_infos_line[part_id2].score + (*skeleton1)[part_id2].score;
                    (*skeleton1)[SCORE_IDX].score += peak_infos_line[cur_conn->peak_id2].score + cur_conn->score;
                    (*skeleton1)[LIMB_INFO_IDX].length = max((*skeleton1)[LIMB_INFO_IDX].length, cur_conn->length);
                }
            } else if (num_found == 2) {
                vector<ConnectedPeak> *skeleton1 = &skeletons[skeleton_idx1];
                vector<ConnectedPeak> *skeleton2 = &skeletons[skeleton_idx2];
                int skeleton1_limb_len = (*skeleton1)[LIMB_INFO_IDX].length * LIMB_LENGTH_RATE;

                // check membership
                bool id1_in_skeleton1 = false;
                for (int kp_id=0; kp_id<NUM_PART; kp_id++)
                    if (cur_conn->peak_id1 == (*skeleton1)[kp_id].id) id1_in_skeleton1 = true;

                bool is_member = false;
                int conn1_idx, conn2_idx;
                float skeleton_limb_min_score = 0.0;
                for (int kp_id = 0; kp_id < NUM_PART; kp_id++) {
                    // used when `is_member` is `false`
                    bool is_sk1_assigned = (*skeleton1)[kp_id].id > 0;
                    bool is_sk2_assigned = (*skeleton2)[kp_id].id > 0;
                    if (is_sk1_assigned)
                        skeleton_limb_min_score = min(skeleton_limb_min_score, (*skeleton1)[kp_id].score);
                    if (is_sk2_assigned)
                        skeleton_limb_min_score = min(skeleton_limb_min_score, (*skeleton2)[kp_id].score);
                    if (is_sk1_assigned && is_sk2_assigned) is_member = true;

                    // used when `is_member` is `true`
                    if (id1_in_skeleton1) {
                        if (cur_conn->peak_id1 == (*skeleton1)[kp_id].id) conn1_idx = kp_id;
                        if (cur_conn->peak_id2 == (*skeleton2)[kp_id].id) conn2_idx = kp_id;
                    } else {
                        if (cur_conn->peak_id1 == (*skeleton2)[kp_id].id) conn1_idx = kp_id;
                        if (cur_conn->peak_id2 == (*skeleton1)[kp_id].id) conn2_idx = kp_id;
                    }
                }
                // If both people have no same joints connected, merge into a single person
                if (!is_member) {
                    if (skeleton_limb_min_score * MIN_SCORE_TOLERANCE > cur_conn->score ||
                        skeleton1_limb_len > cur_conn->length) {
                        // Update which joints are connected
                        for (int kp_id = 0; kp_id < NUM_PART; kp_id++) {
                            (*skeleton1)[kp_id].id += ((*skeleton2)[kp_id].id + 1);
                            (*skeleton1)[kp_id].score += ((*skeleton2)[kp_id].score + 1);
                        }
                        (*skeleton1)[LIMB_INFO_IDX].count += (*skeleton2)[LIMB_INFO_IDX].count;
                        (*skeleton1)[LIMB_INFO_IDX].length = max((*skeleton1)[LIMB_INFO_IDX].length, cur_conn->length);
                        (*skeleton1)[SCORE_IDX].score += (*skeleton2)[SCORE_IDX].score + cur_conn->score;
                        skeletons.erase(skeletons.begin() + skeleton_idx2);
                    }
                } else {
                    assert (conn1_idx != conn2_idx && "Candidate keypoint cannot be shared by 2+ skeletons.");
                    // current limb score is higher than connected ones.
                    if (cur_conn->score >= (*skeleton1)[conn1_idx].score &&
                        cur_conn->score >= (*skeleton2)[conn2_idx].score) {

                        int low_conf_idx, high_conf_idx, remove_conn_idx;
                        if ((*skeleton1)[conn1_idx].score > (*skeleton2)[conn2_idx].score) {
                            low_conf_idx = skeleton_idx2;
                            high_conf_idx = skeleton_idx1;
                            remove_conn_idx = conn2_idx;
                        } else {
                            low_conf_idx = skeleton_idx1;
                            high_conf_idx = skeleton_idx2;
                            remove_conn_idx = conn1_idx;
                        }

                        if (DELETE_SHARED_JOINTS) {
                            vector<ConnectedPeak> * low_conf_skeleton = &skeletons[low_conf_idx];
                            (*low_conf_skeleton)[SCORE_IDX].score -= \
                                peak_infos_line[(*low_conf_skeleton)[remove_conn_idx].id].score +
                                (*low_conf_skeleton)[remove_conn_idx].score;
                            (*low_conf_skeleton)[remove_conn_idx].id = -1;
                            (*low_conf_skeleton)[remove_conn_idx].score = -1;
                            (*low_conf_skeleton)[LIMB_INFO_IDX].count -= 1;
                        }
                    }
                }
            } else if (num_found == 0 && pair_id < NUM_PART) {
                vector<ConnectedPeak> new_skeleton(NUM_PART_OUTS);
                for (int i = 0; i < NUM_PART_OUTS; i++) {  // assign -1 to all
                    new_skeleton[i].id = -1;
                    new_skeleton[i].score = -1;
                }
                new_skeleton[part_id1].id = cur_conn->peak_id1;
                new_skeleton[part_id2].id = cur_conn->peak_id2;
                new_skeleton[part_id1].score = cur_conn->score;
                new_skeleton[part_id2].score = cur_conn->score;
                new_skeleton[SCORE_IDX].score = peak_infos_line[cur_conn->peak_id1].score +
                                                peak_infos_line[cur_conn->peak_id2].score +
                                                cur_conn->score;
                new_skeleton[LIMB_INFO_IDX].count = 2;
                new_skeleton[LIMB_INFO_IDX].length = cur_conn->length;
                skeletons.push_back(new_skeleton);
            }
        }
    }

    // `skeletons`: (#skeletons, 20, 2), delete some rows
    for (int i = skeletons.size() - 1; i >= 0; i--) {
        if (skeletons[i][LIMB_INFO_IDX].count < THRESH_PART_CNT ||
            skeletons[i][SCORE_IDX].score / skeletons[i][LIMB_INFO_IDX].count < THRESH_HUMAN_SCORE)
            skeletons.erase(skeletons.begin() + i);
    }

    return 0;
}

int get_num_humans() {
    return skeletons.size();
}

int get_part_conn_id(int skeleton_id, int part_id) {
    return skeletons[skeleton_id][part_id].id;
}

float get_score(int skeleton_id) {
    return skeletons[skeleton_id][SCORE_IDX].score / skeletons[skeleton_id][LIMB_INFO_IDX].count;
}

int get_part_x(int cid) {
    return peak_infos_line[cid].x;
}

int get_part_y(int cid) {
    return peak_infos_line[cid].y;
}

float get_part_score(int cid) {
    return peak_infos_line[cid].score;
}

vector <float>
get_paf_scores(float *pafmap, int pair_id, int num_steps, int &f2, int &f3, Peak &peak1, Peak &peak2) {
    vector <float> paf_scores;

    const float STEP_X = (peak2.x - peak1.x) / float(num_steps);
    const float STEP_Y = (peak2.y - peak1.y) / float(num_steps);

    for (int i = 0; i < num_steps; i++) {
        int location_x = round2int(peak1.x + i * STEP_X);
        int location_y = round2int(peak1.y + i * STEP_Y);

        float score = PAF(location_y, location_x, pair_id);
        paf_scores.push_back(score);
    }

    return paf_scores;
}

int round2int(float v) {
    return (int) (v + 0.5);
}

bool comp_candidate(ConnectionCandidate a, ConnectionCandidate b) {
    return a.score > b.score;
}
