#include <vector>

#ifndef PAFPROCESS
#define PAFPROCESS

const float THRESH_HEAT = 0.05;
const float THRESH_VECTOR_SCORE = 0.05;
const int THRESH_VECTOR_CNT1 = 6;
const int THRESH_PART_CNT = 4;
const float THRESH_HUMAN_SCORE = 0.3;
const int NUM_PART = 14;

const int STEP_PAF = 10;

const int COCOPAIRS_SIZE = 15;
const int COCOPAIRS_NET[COCOPAIRS_SIZE][2] = {
    {2, 3}, {8, 9}, {4, 5}, {6, 7}, {10, 11}, {12, 13}, {14, 15}, {16, 17}, {18, 19},
    {20, 21}, {22, 23}, {24, 25}, {0, 1}, {26, 27}, {28, 29}
};

const int COCOPAIRS[COCOPAIRS_SIZE][2] = {
    {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10},
    {1, 11}, {11, 12}, {12, 13}, {1, 0}, {2, 0}, {5, 0}
};

struct Peak {
    int x;
    int y;
    float score;
    int id;
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
};

struct Connection {
    int cid1;
    int cid2;
    float score;
    int peak_id1;
    int peak_id2;
};

int process_paf(int p1, int p2, int p3, float *peaks, int h1, int h2, int h3, float *heatmap, int f1, int f2, int f3, float *pafmap);
int get_num_humans();
int get_part_cid(int human_id, int part_id);
float get_score(int human_id);
int get_part_x(int cid);
int get_part_y(int cid);
float get_part_score(int cid);

#endif
