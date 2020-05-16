#include <vector>

#ifndef PAFPROCESS
#define PAFPROCESS

const float THRESH_HEAT = 0.05;
const float THRESH_PAF_SCORE = 0.1;
const float THRESH_PAF_STEP_RATIO = 0.8;
const int THRESH_PART_CNT = 2;
const float THRESH_HUMAN_SCORE = 0.45;
const int NUM_PART = 18;
const int NUM_PART_OUTS = NUM_PART + 2;
const int STEP_PAF = 20;
const int LIMB_LENGTH_RATE = 16;
const float MIN_SCORE_TOLERANCE = 0.7;
const bool DELETE_SHARED_JOINTS = false;

const int COCOPAIRS_SIZE = 30;
const int COCOPAIRS[COCOPAIRS_SIZE][2] = {
        { 1,  0}, { 1, 14}, { 1, 15}, { 1, 16}, { 1, 17}, {0, 14},
        { 0, 15}, {14, 16}, {15, 17}, { 1,  2}, { 2,  3}, {3,  4},
        { 1,  5}, { 5,  6}, { 6,  7}, { 1,  8}, { 8,  9}, {9, 10},
        { 1, 11}, {11, 12}, {12, 13}, { 0,  2}, { 0,  5}, {2,  8},
        { 8, 12}, { 5, 11}, {11,  9}, {16,  2}, {17,  5}, {8, 11}
};

struct Peak {
    int x;
    int y;
    float score;
    int id;
};

struct ConnectedPeak {
    union {
        int id;
        int count;
    };
    union {
        float score;
        float length;
    };
};

struct VectorXY {
    float x;
    float y;
};

struct ConnectionCandidate {
    int idx1;
    int idx2;
    float score;
    float etc;
    float length;
};

struct Connection {
    int cid1;
    int cid2;
    float score;
    int peak_id1;
    int peak_id2;
    float length;
};


int process_paf(int p1, int p2, int p3, float *peaks, int h1, int h2, int h3, float *heatmap, int f1, int f2, int f3, float *pafmap);
int get_num_humans();
int get_part_peak_id(int skeleton_id, int part_id);
float get_score(int skeleton_id);
int get_part_x(int cid);
int get_part_y(int cid);
float get_part_score(int cid);

#endif
