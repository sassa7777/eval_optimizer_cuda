
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <atomic>
#include <bit>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
using namespace std;
using namespace std::chrono;

struct Adam {
    float w = 0;
    float delta = 0;
    float m = 0.0, v = 0.0;
    unsigned int t = 0;

    Adam() = default;
};

struct Data {
    uint64_t p;
    uint64_t o;
    int ans;
};

constexpr int pow3[11] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049 };

enum Position : int {
    INDEX_A1 = 63, INDEX_B1 = 62, INDEX_C1 = 61, INDEX_D1 = 60, INDEX_E1 = 59, INDEX_F1 = 58, INDEX_G1 = 57, INDEX_H1 = 56,
    INDEX_A2 = 55, INDEX_B2 = 54, INDEX_C2 = 53, INDEX_D2 = 52, INDEX_E2 = 51, INDEX_F2 = 50, INDEX_G2 = 49, INDEX_H2 = 48,
    INDEX_A3 = 47, INDEX_B3 = 46, INDEX_C3 = 45, INDEX_D3 = 44, INDEX_E3 = 43, INDEX_F3 = 42, INDEX_G3 = 41, INDEX_H3 = 40,
    INDEX_A4 = 39, INDEX_B4 = 38, INDEX_C4 = 37, INDEX_D4 = 36, INDEX_E4 = 35, INDEX_F4 = 34, INDEX_G4 = 33, INDEX_H4 = 32,
    INDEX_A5 = 31, INDEX_B5 = 30, INDEX_C5 = 29, INDEX_D5 = 28, INDEX_E5 = 27, INDEX_F5 = 26, INDEX_G5 = 25, INDEX_H5 = 24,
    INDEX_A6 = 23, INDEX_B6 = 22, INDEX_C6 = 21, INDEX_D6 = 20, INDEX_E6 = 19, INDEX_F6 = 18, INDEX_G6 = 17, INDEX_H6 = 16,
    INDEX_A7 = 15, INDEX_B7 = 14, INDEX_C7 = 13, INDEX_D7 = 12, INDEX_E7 = 11, INDEX_F7 = 10, INDEX_G7 = 9, INDEX_H7 = 8,
    INDEX_A8 = 7, INDEX_B8 = 6, INDEX_C8 = 5, INDEX_D8 = 4, INDEX_E8 = 3, INDEX_F8 = 2, INDEX_G8 = 1, INDEX_H8 = 0
};

// diagonal8
#define DIAGONAL8_0 INDEX_A1, INDEX_B2, INDEX_C3, INDEX_D4, INDEX_E4, INDEX_D5, INDEX_E5, INDEX_F6, INDEX_G7, INDEX_H8
#define DIAGONAL8_1 INDEX_H1, INDEX_G2, INDEX_F3, INDEX_E4, INDEX_E5, INDEX_D4, INDEX_D5, INDEX_C6, INDEX_B7, INDEX_A8

// diagonal7
#define DIAGONAL7_0 INDEX_B1, INDEX_C2, INDEX_D3, INDEX_E3, INDEX_F3, INDEX_E4, INDEX_F4, INDEX_F5, INDEX_G6, INDEX_H7
#define DIAGONAL7_1 INDEX_H2, INDEX_G3, INDEX_F4, INDEX_F5, INDEX_F6, INDEX_E5, INDEX_E6, INDEX_D6, INDEX_C7, INDEX_B8
#define DIAGONAL7_2 INDEX_G8, INDEX_F7, INDEX_E6, INDEX_D6, INDEX_C6, INDEX_D5, INDEX_C5, INDEX_C4, INDEX_B3, INDEX_A2
#define DIAGONAL7_3 INDEX_A7, INDEX_B6, INDEX_C5, INDEX_C4, INDEX_C3, INDEX_D4, INDEX_D3, INDEX_E3, INDEX_F2, INDEX_G1

// diagonal6
#define DIAGONAL6_0 INDEX_C1, INDEX_D1, INDEX_E1, INDEX_D2, INDEX_E3, INDEX_F4, INDEX_H4, INDEX_G5, INDEX_H5, INDEX_H6
#define DIAGONAL6_1 INDEX_H3, INDEX_H4, INDEX_H5, INDEX_G4, INDEX_F5, INDEX_E6, INDEX_E8, INDEX_D7, INDEX_D8, INDEX_C8
#define DIAGONAL6_2 INDEX_F8, INDEX_E8, INDEX_D8, INDEX_E7, INDEX_D6, INDEX_C5, INDEX_A5, INDEX_B4, INDEX_A4, INDEX_A3
#define DIAGONAL6_3 INDEX_A6, INDEX_A5, INDEX_A4, INDEX_B5, INDEX_C4, INDEX_D3, INDEX_D1, INDEX_E2, INDEX_E1, INDEX_F1

// diagonal5
#define DIAGONAL5_0 INDEX_D1, INDEX_E1, INDEX_E2, INDEX_E3, INDEX_F3, INDEX_E4, INDEX_F4, INDEX_G4, INDEX_H4, INDEX_H5
#define DIAGONAL5_1 INDEX_H4, INDEX_H5, INDEX_G5, INDEX_F5, INDEX_F6, INDEX_E5, INDEX_E6, INDEX_E7, INDEX_E8, INDEX_D8
#define DIAGONAL5_2 INDEX_E8, INDEX_D8, INDEX_D7, INDEX_D6, INDEX_C6, INDEX_D5, INDEX_C5, INDEX_B5, INDEX_A5, INDEX_A4
#define DIAGONAL5_3 INDEX_A5, INDEX_A4, INDEX_B4, INDEX_C4, INDEX_C3, INDEX_D4, INDEX_D3, INDEX_D2, INDEX_D1, INDEX_E1

// edge_2x
#define EDGE_2X_0 INDEX_A1, INDEX_B1, INDEX_C1, INDEX_D1, INDEX_E1, INDEX_F1, INDEX_G1, INDEX_H1, INDEX_B2, INDEX_G2
#define EDGE_2X_1 INDEX_H1, INDEX_H2, INDEX_H3, INDEX_H4, INDEX_H5, INDEX_H6, INDEX_H7, INDEX_H8, INDEX_G2, INDEX_G7
#define EDGE_2X_2 INDEX_H8, INDEX_G8, INDEX_F8, INDEX_E8, INDEX_D8, INDEX_C8, INDEX_B8, INDEX_A8, INDEX_G7, INDEX_B7
#define EDGE_2X_3 INDEX_A8, INDEX_A7, INDEX_A6, INDEX_A5, INDEX_A4, INDEX_A3, INDEX_A2, INDEX_A1, INDEX_B7, INDEX_B2

// h_v_2
#define H_V_2_0 INDEX_A1, INDEX_H1, INDEX_A2, INDEX_B2, INDEX_C2, INDEX_D2, INDEX_E2, INDEX_F2, INDEX_G2, INDEX_H2
#define H_V_2_1 INDEX_H1, INDEX_H8, INDEX_G1, INDEX_G2, INDEX_G3, INDEX_G4, INDEX_G5, INDEX_G6, INDEX_G7, INDEX_G8
#define H_V_2_2 INDEX_H8, INDEX_A8, INDEX_H7, INDEX_G7, INDEX_F7, INDEX_E7, INDEX_D7, INDEX_C7, INDEX_B7, INDEX_A7
#define H_V_2_3 INDEX_A8, INDEX_A1, INDEX_B8, INDEX_B7, INDEX_B6, INDEX_B5, INDEX_B4, INDEX_B3, INDEX_B2, INDEX_B1

// h_v_3
#define H_V_3_0 INDEX_D2, INDEX_E2, INDEX_A3, INDEX_B3, INDEX_C3, INDEX_D3, INDEX_E3, INDEX_F3, INDEX_G3, INDEX_H3
#define H_V_3_1 INDEX_G4, INDEX_G5, INDEX_F1, INDEX_F2, INDEX_F3, INDEX_F4, INDEX_F5, INDEX_F6, INDEX_F7, INDEX_F8
#define H_V_3_2 INDEX_E7, INDEX_D7, INDEX_H6, INDEX_G6, INDEX_F6, INDEX_E6, INDEX_D6, INDEX_C6, INDEX_B6, INDEX_A6
#define H_V_3_3 INDEX_B5, INDEX_B4, INDEX_C8, INDEX_C7, INDEX_C6, INDEX_C5, INDEX_C4, INDEX_C3, INDEX_C2, INDEX_C1

// h_v_4
#define H_V_4_0 INDEX_D3, INDEX_E3, INDEX_A4, INDEX_B4, INDEX_C4, INDEX_D4, INDEX_E4, INDEX_F4, INDEX_G4, INDEX_H4
#define H_V_4_1 INDEX_F4, INDEX_F5, INDEX_E1, INDEX_E2, INDEX_E3, INDEX_E4, INDEX_E5, INDEX_E6, INDEX_E7, INDEX_E8
#define H_V_4_2 INDEX_E6, INDEX_D6, INDEX_H5, INDEX_G5, INDEX_F5, INDEX_E5, INDEX_D5, INDEX_C5, INDEX_B5, INDEX_A5
#define H_V_4_3 INDEX_C5, INDEX_C4, INDEX_D8, INDEX_D7, INDEX_D6, INDEX_D5, INDEX_D4, INDEX_D3, INDEX_D2, INDEX_D1

// corner_3x3
#define CORNER_3X3_0 INDEX_A1, INDEX_B1, INDEX_C1, INDEX_A2, INDEX_B2, INDEX_C2, INDEX_A3, INDEX_B3, INDEX_C3, INDEX_D4
#define CORNER_3X3_1 INDEX_H1, INDEX_H2, INDEX_H3, INDEX_G1, INDEX_G2, INDEX_G3, INDEX_F1, INDEX_F2, INDEX_F3, INDEX_E4
#define CORNER_3X3_2 INDEX_H8, INDEX_G8, INDEX_F8, INDEX_H7, INDEX_G7, INDEX_F7, INDEX_H6, INDEX_G6, INDEX_F6, INDEX_E5
#define CORNER_3X3_3 INDEX_A8, INDEX_A7, INDEX_A6, INDEX_B8, INDEX_B7, INDEX_B6, INDEX_C8, INDEX_C7, INDEX_C6, INDEX_D5

// edge_x_side
#define EDGE_X_SIDE_0 INDEX_A1, INDEX_B1, INDEX_C1, INDEX_D1, INDEX_E1, INDEX_A2, INDEX_B2, INDEX_A3, INDEX_A4, INDEX_A5
#define EDGE_X_SIDE_1 INDEX_H1, INDEX_H2, INDEX_H3, INDEX_H4, INDEX_H5, INDEX_G1, INDEX_G2, INDEX_F1, INDEX_E1, INDEX_D1
#define EDGE_X_SIDE_2 INDEX_H8, INDEX_G8, INDEX_F8, INDEX_E8, INDEX_D8, INDEX_H7, INDEX_G7, INDEX_H6, INDEX_H5, INDEX_H4
#define EDGE_X_SIDE_3 INDEX_A8, INDEX_A7, INDEX_A6, INDEX_A5, INDEX_A4, INDEX_B8, INDEX_B7, INDEX_C8, INDEX_D8, INDEX_E8

// edge_block
#define EDGE_BLOCK_0 INDEX_A1, INDEX_C1, INDEX_D1, INDEX_E1, INDEX_F1, INDEX_H1, INDEX_C2, INDEX_D2, INDEX_E2, INDEX_F2
#define EDGE_BLOCK_1 INDEX_H1, INDEX_H3, INDEX_H4, INDEX_H5, INDEX_H6, INDEX_H8, INDEX_G3, INDEX_G4, INDEX_G5, INDEX_G6
#define EDGE_BLOCK_2 INDEX_H8, INDEX_F8, INDEX_E8, INDEX_D8, INDEX_C8, INDEX_A8, INDEX_F7, INDEX_E7, INDEX_D7, INDEX_C7
#define EDGE_BLOCK_3 INDEX_A8, INDEX_A6, INDEX_A5, INDEX_A4, INDEX_A3, INDEX_A1, INDEX_B6, INDEX_B5, INDEX_B4, INDEX_B3

// triangle
#define TRIANGLE_0 INDEX_A1, INDEX_B1, INDEX_C1, INDEX_D1, INDEX_A2, INDEX_B2, INDEX_C2, INDEX_A3, INDEX_B3, INDEX_A4
#define TRIANGLE_1 INDEX_H1, INDEX_H2, INDEX_H3, INDEX_H4, INDEX_G1, INDEX_G2, INDEX_G3, INDEX_F1, INDEX_F2, INDEX_E1
#define TRIANGLE_2 INDEX_H8, INDEX_G8, INDEX_F8, INDEX_E8, INDEX_H7, INDEX_G7, INDEX_F7, INDEX_H6, INDEX_G6, INDEX_H5
#define TRIANGLE_3 INDEX_A8, INDEX_A7, INDEX_A6, INDEX_A5, INDEX_B8, INDEX_B7, INDEX_B6, INDEX_C8, INDEX_C7, INDEX_D8

// corner_2x5
#define CORNER_2X5_0 INDEX_A1, INDEX_B1, INDEX_C1, INDEX_D1, INDEX_E1, INDEX_A2, INDEX_B2, INDEX_C2, INDEX_D2, INDEX_E2
#define CORNER_2X5_1 INDEX_H1, INDEX_H2, INDEX_H3, INDEX_H4, INDEX_H5, INDEX_G1, INDEX_G2, INDEX_G3, INDEX_G4, INDEX_G5
#define CORNER_2X5_2 INDEX_H8, INDEX_G8, INDEX_F8, INDEX_E8, INDEX_D8, INDEX_H7, INDEX_G7, INDEX_F7, INDEX_E7, INDEX_D7
#define CORNER_2X5_3 INDEX_A8, INDEX_A7, INDEX_A6, INDEX_A5, INDEX_A4, INDEX_B8, INDEX_B7, INDEX_B6, INDEX_B5, INDEX_B4

#define CORNER_2X5_4 INDEX_A1, INDEX_A2, INDEX_A3, INDEX_A4, INDEX_A5, INDEX_B1, INDEX_B2, INDEX_B3, INDEX_B4, INDEX_B5
#define CORNER_2X5_5 INDEX_H1, INDEX_G1, INDEX_F1, INDEX_E1, INDEX_D1, INDEX_H2, INDEX_G2, INDEX_F2, INDEX_E2, INDEX_D2
#define CORNER_2X5_6 INDEX_H8, INDEX_H7, INDEX_H6, INDEX_H5, INDEX_H4, INDEX_G8, INDEX_G7, INDEX_G6, INDEX_G5, INDEX_G4
#define CORNER_2X5_7 INDEX_A8, INDEX_B8, INDEX_C8, INDEX_D8, INDEX_E8, INDEX_A7, INDEX_B7, INDEX_C7, INDEX_D7, INDEX_E7



__device__ __constant__ int POW3[11] = { 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049 };

// 5ビット
__device__ int CALC_INDEX(const uint64_t B, const int t5, const int t6, const int t7, const int t8, const int t9) {
    return ((B >> t5) & 1) * POW3[4] +
        ((B >> t6) & 1) * POW3[3] +
        ((B >> t7) & 1) * POW3[2] +
        ((B >> t8) & 1) * POW3[1] +
        ((B >> t9) & 1) * POW3[0];
}

// 6ビット
__device__ int CALC_INDEX(const uint64_t B, const int t4, const int t5, const int t6, const int t7, const int t8, const int t9) {
    return ((B >> t4) & 1) * POW3[5] +
        ((B >> t5) & 1) * POW3[4] +
        ((B >> t6) & 1) * POW3[3] +
        ((B >> t7) & 1) * POW3[2] +
        ((B >> t8) & 1) * POW3[1] +
        ((B >> t9) & 1) * POW3[0];
}

// 7ビット
__device__ int CALC_INDEX(const uint64_t B, const int t3, const int t4, const int t5, const int t6, const int t7, const int t8, const int t9) {
    return ((B >> t3) & 1) * POW3[6] +
        ((B >> t4) & 1) * POW3[5] +
        ((B >> t5) & 1) * POW3[4] +
        ((B >> t6) & 1) * POW3[3] +
        ((B >> t7) & 1) * POW3[2] +
        ((B >> t8) & 1) * POW3[1] +
        ((B >> t9) & 1) * POW3[0];
}

// 8ビット
__device__ int CALC_INDEX(const uint64_t B, const int t2, const int t3, const int t4, const int t5, const int t6, const int t7, const int t8, const int t9) {
    return ((B >> t2) & 1) * POW3[7] +
        ((B >> t3) & 1) * POW3[6] +
        ((B >> t4) & 1) * POW3[5] +
        ((B >> t5) & 1) * POW3[4] +
        ((B >> t6) & 1) * POW3[3] +
        ((B >> t7) & 1) * POW3[2] +
        ((B >> t8) & 1) * POW3[1] +
        ((B >> t9) & 1) * POW3[0];
}

// 9ビット
__device__ int CALC_INDEX(const uint64_t B, const int t1, const int t2, const int t3, const int t4, const int t5, const int t6, const int t7, const int t8, const int t9) {
    return ((B >> t1) & 1) * POW3[8] +
        ((B >> t2) & 1) * POW3[7] +
        ((B >> t3) & 1) * POW3[6] +
        ((B >> t4) & 1) * POW3[5] +
        ((B >> t5) & 1) * POW3[4] +
        ((B >> t6) & 1) * POW3[3] +
        ((B >> t7) & 1) * POW3[2] +
        ((B >> t8) & 1) * POW3[1] +
        ((B >> t9) & 1) * POW3[0];
}

// 10ビット
__device__ int CALC_INDEX(const uint64_t B, const int t0, const int t1, const int t2, const int t3, const int t4, const int t5, const int t6, const int t7, const int t8, const int t9) {
    return ((B >> t0) & 1) * POW3[9] +
        ((B >> t1) & 1) * POW3[8] +
        ((B >> t2) & 1) * POW3[7] +
        ((B >> t3) & 1) * POW3[6] +
        ((B >> t4) & 1) * POW3[5] +
        ((B >> t5) & 1) * POW3[4] +
        ((B >> t6) & 1) * POW3[3] +
        ((B >> t7) & 1) * POW3[2] +
        ((B >> t8) & 1) * POW3[1] +
        ((B >> t9) & 1) * POW3[0];
}
class Pattern_Eval {
public:
    Adam diagonal8[pow3[10]];
    Adam diagonal7[pow3[10]];
    Adam diagonal6[pow3[10]];
    Adam diagonal5[pow3[10]];
    Adam edge_2x[pow3[10]];
    Adam h_v_2[pow3[10]];
    Adam h_v_3[pow3[10]];
    Adam h_v_4[pow3[10]];
    Adam corner_3x3[pow3[10]];
    Adam edge_x_side[pow3[10]];
    Adam edge_block[pow3[10]];
    Adam triangle[pow3[10]];
    Adam corner_2x5[pow3[10]];
};

Adam mobility_arr[36 * 36];
Adam stone_arr[65 * 65];

// code from http://www.amy.hi-ho.ne.jp/okuhara/bitboard.htm
__device__ uint64_t makelegalboard(uint64_t p, uint64_t o) {
    uint64_t moves, hb, flip1, flip7, flip9, flip8, pre1, pre7, pre9, pre8;

    hb = o & 0x7e7e7e7e7e7e7e7eULL;
    flip1 = hb & (p << 1);    flip7 = hb & (p << 7);        flip9 = hb & (p << 9);        flip8 = o & (p << 8);
    flip1 |= hb & (flip1 << 1);    flip7 |= hb & (flip7 << 7);    flip9 |= hb & (flip9 << 9);    flip8 |= o & (flip8 << 8);
    pre1 = hb & (hb << 1);         pre7 = hb & (hb << 7);        pre9 = hb & (hb << 9);        pre8 = o & (o << 8);
    flip1 |= pre1 & (flip1 << 2);    flip7 |= pre7 & (flip7 << 14);    flip9 |= pre9 & (flip9 << 18);    flip8 |= pre8 & (flip8 << 16);
    flip1 |= pre1 & (flip1 << 2);    flip7 |= pre7 & (flip7 << 14);    flip9 |= pre9 & (flip9 << 18);    flip8 |= pre8 & (flip8 << 16);
    moves = flip1 << 1;        moves |= flip7 << 7;        moves |= flip9 << 9;        moves |= flip8 << 8;
    flip1 = hb & (p >> 1);        flip7 = hb & (p >> 7);        flip9 = hb & (p >> 9);        flip8 = o & (p >> 8);
    flip1 |= hb & (flip1 >> 1);    flip7 |= hb & (flip7 >> 7);    flip9 |= hb & (flip9 >> 9);    flip8 |= o & (flip8 >> 8);
    pre1 >>= 1;            pre7 >>= 7;            pre9 >>= 9;            pre8 >>= 8;
    flip1 |= pre1 & (flip1 >> 2);    flip7 |= pre7 & (flip7 >> 14);    flip9 |= pre9 & (flip9 >> 18);    flip8 |= pre8 & (flip8 >> 16);
    flip1 |= pre1 & (flip1 >> 2);    flip7 |= pre7 & (flip7 >> 14);    flip9 |= pre9 & (flip9 >> 18);    flip8 |= pre8 & (flip8 >> 16);
    moves |= flip1 >> 1;        moves |= flip7 >> 7;        moves |= flip9 >> 9;        moves |= flip8 >> 8;

    return moves & ~(p | o);
}

inline uint64_t delta_swap(uint64_t x, uint64_t mask, int delta) {
    uint64_t t = (x ^ (x >> delta)) & mask;
    return x ^ t ^ (t << delta);
}
inline uint64_t flipVertical(uint64_t x) {
    x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x << 8) & 0xFF00FF00FF00FF00ULL);
    x = ((x >> 16) & 0x0000FFFF0000FFFFULL) | ((x << 16) & 0xFFFF0000FFFF0000ULL);
    return ((x >> 32) & 0x00000000FFFFFFFFULL) | ((x << 32) & 0xFFFFFFFF00000000ULL);
}

inline uint64_t flipHorizontal(uint64_t x) {
    x = ((x >> 1) & 0x5555555555555555) | ((x & 0x5555555555555555) << 1);
    x = ((x >> 2) & 0x3333333333333333) | ((x & 0x3333333333333333) << 2);
    x = ((x >> 4) & 0x0f0f0f0f0f0f0f0f) | ((x & 0x0f0f0f0f0f0f0f0f) << 4);
    return x;
}

inline uint64_t flipDiagonalA1H8(uint64_t x) {
    x = delta_swap(x, 0x00AA00AA00AA00AA, 7);
    x = delta_swap(x, 0x0000CCCC0000CCCC, 14);
    return delta_swap(x, 0x00000000F0F0F0F0, 28);
}

inline uint64_t flipDiagonalA8H1(uint64_t x) {
    x = delta_swap(x, 0x0055005500550055, 9);
    x = delta_swap(x, 0x0000333300003333, 18);
    return delta_swap(x, 0x000000000F0F0F0F, 36);
}


__global__ void init_appear(Pattern_Eval *pattern_arr, Adam* mobility_arr, Adam* stone_arr, const Data* data, size_t size) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    uint64_t P = data[idx].p;
    uint64_t O = data[idx].o;
    atomicAdd((&pattern_arr->diagonal8[CALC_INDEX(P, DIAGONAL8_0) + CALC_INDEX(O, DIAGONAL8_0) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal8[CALC_INDEX(P, DIAGONAL8_1) + CALC_INDEX(O, DIAGONAL8_1) * 2].t), 1);

    atomicAdd((&pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_0) +
        CALC_INDEX(O, DIAGONAL7_0) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_1) +
        CALC_INDEX(O, DIAGONAL7_1) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_2) +
        CALC_INDEX(O, DIAGONAL7_2) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_3) +
        CALC_INDEX(O, DIAGONAL7_3) * 2].t), 1);

    atomicAdd((&pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_0) +
        CALC_INDEX(O, DIAGONAL6_0) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_1) +
        CALC_INDEX(O, DIAGONAL6_1) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_2) +
        CALC_INDEX(O, DIAGONAL6_2) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_3) +
        CALC_INDEX(O, DIAGONAL6_3) * 2].t), 1);

    atomicAdd((&pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_0) +
        CALC_INDEX(O, DIAGONAL5_0) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_1) +
        CALC_INDEX(O, DIAGONAL5_1) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_2) +
        CALC_INDEX(O, DIAGONAL5_2) * 2].t), 1);
    atomicAdd((&pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_3) +
        CALC_INDEX(O, DIAGONAL5_3) * 2].t), 1);

    atomicAdd((&pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_0) + CALC_INDEX(O, EDGE_2X_0) * 2].t), 1);
    atomicAdd((&pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_1) + CALC_INDEX(O, EDGE_2X_1) * 2].t), 1);
    atomicAdd((&pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_2) + CALC_INDEX(O, EDGE_2X_2) * 2].t), 1);
    atomicAdd((&pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_3) + CALC_INDEX(O, EDGE_2X_3) * 2].t), 1);

    atomicAdd((&pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_0) + CALC_INDEX(O, H_V_2_0) * 2].t), 1);
    atomicAdd((&pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_1) + CALC_INDEX(O, H_V_2_1) * 2].t), 1);
    atomicAdd((&pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_2) + CALC_INDEX(O, H_V_2_2) * 2].t), 1);
    atomicAdd((&pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_3) + CALC_INDEX(O, H_V_2_3) * 2].t), 1);

    atomicAdd((&pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_0) + CALC_INDEX(O, H_V_3_0) * 2].t), 1);
    atomicAdd((&pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_1) + CALC_INDEX(O, H_V_3_1) * 2].t), 1);
    atomicAdd((&pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_2) + CALC_INDEX(O, H_V_3_2) * 2].t), 1);
    atomicAdd((&pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_3) + CALC_INDEX(O, H_V_3_3) * 2].t), 1);

    atomicAdd((&pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_0) + CALC_INDEX(O, H_V_4_0) * 2].t), 1);
    atomicAdd((&pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_1) + CALC_INDEX(O, H_V_4_1) * 2].t), 1);
    atomicAdd((&pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_2) + CALC_INDEX(O, H_V_4_2) * 2].t), 1);
    atomicAdd((&pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_3) + CALC_INDEX(O, H_V_4_3) * 2].t), 1);

    atomicAdd((&pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_0) + CALC_INDEX(O, CORNER_3X3_0) * 2].t), 1);
    atomicAdd((&pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_1) + CALC_INDEX(O, CORNER_3X3_1) * 2].t), 1);
    atomicAdd((&pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_2) + CALC_INDEX(O, CORNER_3X3_2) * 2].t), 1);
    atomicAdd((&pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_3) + CALC_INDEX(O, CORNER_3X3_3) * 2].t), 1);

    atomicAdd((&pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_0) + CALC_INDEX(O, EDGE_X_SIDE_0) * 2].t), 1);
    atomicAdd((&pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_1) + CALC_INDEX(O, EDGE_X_SIDE_1) * 2].t), 1);
    atomicAdd((&pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_2) + CALC_INDEX(O, EDGE_X_SIDE_2) * 2].t), 1);
    atomicAdd((&pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_3) + CALC_INDEX(O, EDGE_X_SIDE_3) * 2].t), 1);

    atomicAdd((&pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_0) + CALC_INDEX(O, EDGE_BLOCK_0) * 2].t), 1);
    atomicAdd((&pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_1) + CALC_INDEX(O, EDGE_BLOCK_1) * 2].t), 1);
    atomicAdd((&pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_2) + CALC_INDEX(O, EDGE_BLOCK_2) * 2].t), 1);
    atomicAdd((&pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_3) + CALC_INDEX(O, EDGE_BLOCK_3) * 2].t), 1);

    atomicAdd((&pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_0) + CALC_INDEX(O, TRIANGLE_0) * 2].t), 1);
    atomicAdd((&pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_1) + CALC_INDEX(O, TRIANGLE_1) * 2].t), 1);
    atomicAdd((&pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_2) + CALC_INDEX(O, TRIANGLE_2) * 2].t), 1);
    atomicAdd((&pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_3) + CALC_INDEX(O, TRIANGLE_3) * 2].t), 1);

    atomicAdd((&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_0) + CALC_INDEX(O, CORNER_2X5_0) * 2].t), 1);
    atomicAdd((&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_1) + CALC_INDEX(O, CORNER_2X5_1) * 2].t), 1);
    atomicAdd((&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_2) + CALC_INDEX(O, CORNER_2X5_2) * 2].t), 1);
    atomicAdd((&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_3) + CALC_INDEX(O, CORNER_2X5_3) * 2].t), 1);

    atomicAdd((&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_4) + CALC_INDEX(O, CORNER_2X5_4) * 2].t), 1);
    atomicAdd((&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_5) + CALC_INDEX(O, CORNER_2X5_5) * 2].t), 1);
    atomicAdd((&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_6) + CALC_INDEX(O, CORNER_2X5_6) * 2].t), 1);
    atomicAdd((&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_7) + CALC_INDEX(O, CORNER_2X5_7) * 2].t), 1);

    atomicAdd(&mobility_arr[__popcll(makelegalboard(P, O)) * 36 + __popcll(makelegalboard(O, P))].t, 1);

    //atomicAdd(&stone_arr[__popcll(P) * 65 + __popcll(O)].t, 1);

}

__device__ float evaluate(const uint64_t P, const uint64_t O, Pattern_Eval *pattern_arr, Adam* mobility_arr, Adam* stone_arr) {



    float a = 0;

    a += pattern_arr->diagonal8[CALC_INDEX(P, DIAGONAL8_0) + CALC_INDEX(O, DIAGONAL8_0) * 2].w;
    a += pattern_arr->diagonal8[CALC_INDEX(P, DIAGONAL8_1) + CALC_INDEX(O, DIAGONAL8_1) * 2].w;

    a += pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_0) + CALC_INDEX(O, DIAGONAL7_0) * 2].w;
    a += pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_1) + CALC_INDEX(O, DIAGONAL7_1) * 2].w;
    a += pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_2) + CALC_INDEX(O, DIAGONAL7_2) * 2].w;
    a += pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_3) + CALC_INDEX(O, DIAGONAL7_3) * 2].w;

    a += pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_0) + CALC_INDEX(O, DIAGONAL6_0) * 2].w;
    a += pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_1) + CALC_INDEX(O, DIAGONAL6_1) * 2].w;
    a += pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_2) + CALC_INDEX(O, DIAGONAL6_2) * 2].w;
    a += pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_3) + CALC_INDEX(O, DIAGONAL6_3) * 2].w;

    a += pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_0) + CALC_INDEX(O, DIAGONAL5_0) * 2].w;
    a += pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_1) + CALC_INDEX(O, DIAGONAL5_1) * 2].w;
    a += pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_2) + CALC_INDEX(O, DIAGONAL5_2) * 2].w;
    a += pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_3) + CALC_INDEX(O, DIAGONAL5_3) * 2].w;

    a += pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_0) + CALC_INDEX(O, EDGE_2X_0) * 2].w;
    a += pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_1) + CALC_INDEX(O, EDGE_2X_1) * 2].w;
    a += pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_2) + CALC_INDEX(O, EDGE_2X_2) * 2].w;
    a += pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_3) + CALC_INDEX(O, EDGE_2X_3) * 2].w;

    a += pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_0) + CALC_INDEX(O, H_V_2_0) * 2].w;
    a += pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_1) + CALC_INDEX(O, H_V_2_1) * 2].w;
    a += pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_2) + CALC_INDEX(O, H_V_2_2) * 2].w;
    a += pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_3) + CALC_INDEX(O, H_V_2_3) * 2].w;

    a += pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_0) + CALC_INDEX(O, H_V_3_0) * 2].w;
    a += pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_1) + CALC_INDEX(O, H_V_3_1) * 2].w;
    a += pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_2) + CALC_INDEX(O, H_V_3_2) * 2].w;
    a += pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_3) + CALC_INDEX(O, H_V_3_3) * 2].w;

    a += pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_0) + CALC_INDEX(O, H_V_4_0) * 2].w;
    a += pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_1) + CALC_INDEX(O, H_V_4_1) * 2].w;
    a += pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_2) + CALC_INDEX(O, H_V_4_2) * 2].w;
    a += pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_3) + CALC_INDEX(O, H_V_4_3) * 2].w;

    a += pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_0) + CALC_INDEX(O, CORNER_3X3_0) * 2].w;
    a += pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_1) + CALC_INDEX(O, CORNER_3X3_1) * 2].w;
    a += pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_2) + CALC_INDEX(O, CORNER_3X3_2) * 2].w;
    a += pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_3) + CALC_INDEX(O, CORNER_3X3_3) * 2].w;

    a += pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_0) + CALC_INDEX(O, EDGE_X_SIDE_0) * 2].w;
    a += pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_1) + CALC_INDEX(O, EDGE_X_SIDE_1) * 2].w;
    a += pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_2) + CALC_INDEX(O, EDGE_X_SIDE_2) * 2].w;
    a += pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_3) + CALC_INDEX(O, EDGE_X_SIDE_3) * 2].w;

    a += pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_0) + CALC_INDEX(O, EDGE_BLOCK_0) * 2].w;
    a += pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_1) + CALC_INDEX(O, EDGE_BLOCK_1) * 2].w;
    a += pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_2) + CALC_INDEX(O, EDGE_BLOCK_2) * 2].w;
    a += pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_3) + CALC_INDEX(O, EDGE_BLOCK_3) * 2].w;

    a += pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_0) + CALC_INDEX(O, TRIANGLE_0) * 2].w;
    a += pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_1) + CALC_INDEX(O, TRIANGLE_1) * 2].w;
    a += pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_2) + CALC_INDEX(O, TRIANGLE_2) * 2].w;
    a += pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_3) + CALC_INDEX(O, TRIANGLE_3) * 2].w;

    a += pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_0) + CALC_INDEX(O, CORNER_2X5_0) * 2].w;
    a += pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_1) + CALC_INDEX(O, CORNER_2X5_1) * 2].w;
    a += pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_2) + CALC_INDEX(O, CORNER_2X5_2) * 2].w;
    a += pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_3) + CALC_INDEX(O, CORNER_2X5_3) * 2].w;

    a += pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_4) + CALC_INDEX(O, CORNER_2X5_4) * 2].w;
    a += pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_5) + CALC_INDEX(O, CORNER_2X5_5) * 2].w;
    a += pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_6) + CALC_INDEX(O, CORNER_2X5_6) * 2].w;
    a += pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_7) + CALC_INDEX(O, CORNER_2X5_7) * 2].w;

    a += mobility_arr[__popcll(makelegalboard(P, O)) * 36 + __popcll(makelegalboard(O, P))].w;

    //a += stone_arr[__popcll(P) * 65 + __popcll(O)].w;

    return a;
}

__global__ void cal_diff_test(Pattern_Eval* pattern_arr, Adam* mobility_arr, Adam* stone_arr, Data* data, double* grad_global, double* MSE_global, size_t size) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    uint64_t P = data[idx].p;
    uint64_t O = data[idx].o;
    int ans = data[idx].ans;
    double grad = (ans - evaluate(P, O, pattern_arr, mobility_arr, stone_arr));
    atomicAdd(grad_global, fabsf(grad / 256));
    atomicAdd(MSE_global, (grad / 256) * (grad / 256));
}

__global__ void cal_eval_diff(Pattern_Eval *pattern_arr, Adam *mobility_arr, Adam *stone_arr, Data* data, double*grad_global, double*MSE_global, size_t size) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    uint64_t P = data[idx].p;
    uint64_t O = data[idx].o;
    int ans = data[idx].ans;
    double grad = (ans - evaluate(P, O, pattern_arr, mobility_arr, stone_arr));
    atomicAdd(grad_global, fabs(grad/256));
    atomicAdd(MSE_global, (grad /256 ) * (grad/ 256));



    atomicAdd(&pattern_arr->diagonal8[CALC_INDEX(P, DIAGONAL8_0) + CALC_INDEX(O, DIAGONAL8_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal8[CALC_INDEX(P, DIAGONAL8_1) + CALC_INDEX(O, DIAGONAL8_1) * 2].delta, grad);

    atomicAdd(&pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_0) + CALC_INDEX(O, DIAGONAL7_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_1) + CALC_INDEX(O, DIAGONAL7_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_2) + CALC_INDEX(O, DIAGONAL7_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal7[CALC_INDEX(P, DIAGONAL7_3) + CALC_INDEX(O, DIAGONAL7_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_0) + CALC_INDEX(O, DIAGONAL6_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_1) + CALC_INDEX(O, DIAGONAL6_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_2) + CALC_INDEX(O, DIAGONAL6_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal6[CALC_INDEX(P, DIAGONAL6_3) + CALC_INDEX(O, DIAGONAL6_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_0) + CALC_INDEX(O, DIAGONAL5_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_1) + CALC_INDEX(O, DIAGONAL5_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_2) + CALC_INDEX(O, DIAGONAL5_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->diagonal5[CALC_INDEX(P, DIAGONAL5_3) + CALC_INDEX(O, DIAGONAL5_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_0) + CALC_INDEX(O, EDGE_2X_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_1) + CALC_INDEX(O, EDGE_2X_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_2) + CALC_INDEX(O, EDGE_2X_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->edge_2x[CALC_INDEX(P, EDGE_2X_3) + CALC_INDEX(O, EDGE_2X_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_0) + CALC_INDEX(O, H_V_2_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_1) + CALC_INDEX(O, H_V_2_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_2) + CALC_INDEX(O, H_V_2_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->h_v_2[CALC_INDEX(P, H_V_2_3) + CALC_INDEX(O, H_V_2_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_0) + CALC_INDEX(O, H_V_3_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_1) + CALC_INDEX(O, H_V_3_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_2) + CALC_INDEX(O, H_V_3_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->h_v_3[CALC_INDEX(P, H_V_3_3) + CALC_INDEX(O, H_V_3_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_0) + CALC_INDEX(O, H_V_4_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_1) + CALC_INDEX(O, H_V_4_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_2) + CALC_INDEX(O, H_V_4_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->h_v_4[CALC_INDEX(P, H_V_4_3) + CALC_INDEX(O, H_V_4_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_0) + CALC_INDEX(O, CORNER_3X3_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_1) + CALC_INDEX(O, CORNER_3X3_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_2) + CALC_INDEX(O, CORNER_3X3_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->corner_3x3[CALC_INDEX(P, CORNER_3X3_3) + CALC_INDEX(O, CORNER_3X3_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_0) + CALC_INDEX(O, EDGE_X_SIDE_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_1) + CALC_INDEX(O, EDGE_X_SIDE_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_2) + CALC_INDEX(O, EDGE_X_SIDE_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->edge_x_side[CALC_INDEX(P, EDGE_X_SIDE_3) + CALC_INDEX(O, EDGE_X_SIDE_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_0) + CALC_INDEX(O, EDGE_BLOCK_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_1) + CALC_INDEX(O, EDGE_BLOCK_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_2) + CALC_INDEX(O, EDGE_BLOCK_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->edge_block[CALC_INDEX(P, EDGE_BLOCK_3) + CALC_INDEX(O, EDGE_BLOCK_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_0) + CALC_INDEX(O, TRIANGLE_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_1) + CALC_INDEX(O, TRIANGLE_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_2) + CALC_INDEX(O, TRIANGLE_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->triangle[CALC_INDEX(P, TRIANGLE_3) + CALC_INDEX(O, TRIANGLE_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_0) + CALC_INDEX(O, CORNER_2X5_0) * 2].delta, grad);
    atomicAdd(&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_1) + CALC_INDEX(O, CORNER_2X5_1) * 2].delta, grad);
    atomicAdd(&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_2) + CALC_INDEX(O, CORNER_2X5_2) * 2].delta, grad);
    atomicAdd(&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_3) + CALC_INDEX(O, CORNER_2X5_3) * 2].delta, grad);

    atomicAdd(&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_4) + CALC_INDEX(O, CORNER_2X5_4) * 2].delta, grad);
    atomicAdd(&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_5) + CALC_INDEX(O, CORNER_2X5_5) * 2].delta, grad);
    atomicAdd(&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_6) + CALC_INDEX(O, CORNER_2X5_6) * 2].delta, grad);
    atomicAdd(&pattern_arr->corner_2x5[CALC_INDEX(P, CORNER_2X5_7) + CALC_INDEX(O, CORNER_2X5_7) * 2].delta, grad);

    atomicAdd(&mobility_arr[__popcll(makelegalboard(P, O)) * 36 + __popcll(makelegalboard(O, P))].delta, grad);

    //atomicAdd(&stone_arr[__popcll(P) * 65 + __popcll(O)].delta, grad);

}

//__global__ void adam(Adam* a, double alpha_t, unsigned int phase, int num_elements) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (idx >= num_elements) return;  // 範囲外のインデックスに対しては何もしない
//
//    constexpr double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
//
//    if (a[idx].t == 0 || a[idx].delta == 0) return;
//
//    // 学習率の計算
//    double lr = alpha_t / a[idx].t;
//    double grad = a[idx].delta * 2;  // 勾配の計算
//    double lrt = lr * sqrt(1.0 - pow(beta2, phase)) / (1.0 - pow(beta1, phase));
//
//    // モーメントの更新
//    a[idx].m += (1.0 - beta1) * (grad - a[idx].m);
//    a[idx].v += (1.0 - beta2) * (grad * grad - a[idx].v);
//
//    // パラメータ更新
//    double update = lrt * a[idx].m / (sqrt(a[idx].v) + epsilon);
//    a[idx].w += update;
//
//    // 勾配をゼロにリセット
//    a[idx].delta = 0;
//}

//__global__ void gd(Adam* a, double alpha_t, unsigned int phase, int num_elements) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (idx >= num_elements) return;  // 範囲外のインデックスに対しては何もしない
//
//    if (a[idx].t == 0 || a[idx].delta == 0) return;
//
//    //// 学習率の計算
//    //double lr = alpha_t / a[idx].t;
//    //double grad = a[idx].delta * 2;  // 勾配の計算
//
//    a[idx].w += a[idx].delta * 2 / a[idx].t * alpha_t;
//
//    // 勾配をゼロにリセット
//    a[idx].delta = 0;
//}

void gd(Adam& a, double alpha_t, size_t phase) {
    if (a.t == 0) return;
    constexpr float lambda = 0.001;
    constexpr float MAX_NUM = 3840;
    a.w += a.delta * 2 / a.t * alpha_t - a.w * lambda;
    a.w = min(max(a.w, -MAX_NUM), MAX_NUM);
    a.delta = 0;
}

void adam(Adam& a, double alpha_t, size_t phase) {
    if (a.t == 0) return;
    constexpr float MAX_NUM = 5120;
    constexpr double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, lambda = 0.001;
    double lr = alpha_t;
    double grad = a.delta * 2 / a.t - 2 * a.w * lambda;  // 勾配の計算
    double lrt = lr * sqrt(1.0 - pow(beta2, phase)) / (1.0 - pow(beta1, phase));
    //    // モーメントの更新
    a.m += (1.0 - beta1) * (grad - a.m);
    a.v += (1.0 - beta2) * (grad * grad - a.v);

    // パラメータ更新
    double update = lrt * a.m / (sqrt(a.v) + epsilon);
    a.w += update;
    a.w = min(max(a.w, -MAX_NUM), MAX_NUM);

    // 勾配をゼロにリセット
    a.delta = 0;
}



inline void fastParseLine(const std::string& line, uint64_t& p, uint64_t& o, int& ans) {
    const char* s = line.c_str();
    p = 0;
    o = 0;
    ans = 0;

    // p のパース
    while (*s >= '0' && *s <= '9') {
        p = p * 10 + (*s - '0');
        s++;
    }
    while (*s == ' ') s++;  // 空白をスキップ

    // o のパース
    while (*s >= '0' && *s <= '9') {
        o = o * 10 + (*s - '0');
        s++;
    }
    while (*s == ' ') s++;  // 空白をスキップ

    // ans のパース（符号付き）
    bool neg = false;
    if (*s == '-') {
        neg = true;
        s++;
    }
    while (*s >= '0' && *s <= '9') {
        ans = ans * 10 + (*s - '0');
        s++;
    }
    if (neg) ans = -ans;
}



Pattern_Eval pattern_arr;

int popcount(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555);
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x * 0x0101010101010101) >> 56;
    return x;
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << "start" << endl;
    int PHASE = atoi(argv[1]);
    //int PHASE = 0;
    int game_phase_start = PHASE * 4 + 1 + 4 - 2;
    int game_phase_end = (PHASE + 1) * 4 + 4 + 2;
    double alpha = 0.005;
    // double alpha = 50;
    //double alpha_t = alpha / 5.0;
    double alpha_t = alpha;
    size_t train_c, test_c;

    vector<Data> train_data_vec, test_data_vec;
    ifstream train_file("train_data.txt");
    string line;
    train_data_vec.reserve(10000000);
    while (getline(train_file, line)) {
        uint64_t p, o;
        int ans;
        fastParseLine(line, p, o, ans);
        if (!(popcount(p | o) >= game_phase_start && popcount(p | o) <= game_phase_end)) continue;
        Data d;
        d.p = p;
        d.o = o;
        d.ans = ans;
        train_data_vec.emplace_back(d);
        d.p = flipHorizontal(p);
        d.o = flipHorizontal(o);
        train_data_vec.emplace_back(d);
        d.p = o;
        d.o = p;
        d.ans = -ans;
        train_data_vec.emplace_back(d);
        d.p = flipHorizontal(o);
        d.o = flipHorizontal(p);
        train_data_vec.emplace_back(d);
    }
    train_data_vec.shrink_to_fit();
    ifstream test_file("test_data.txt");
    while (getline(test_file, line)) {
        uint64_t p, o;
        int ans;
        fastParseLine(line, p, o, ans);
        if (!(popcount(p | o) >= game_phase_start && popcount(p | o) <= game_phase_end)) continue;
        Data d;
        d.p = p;
        d.o = o;
        d.ans = ans;
        test_data_vec.emplace_back(d);
        d.p = flipHorizontal(p);
        d.o = flipHorizontal(o);
        test_data_vec.emplace_back(d);
        d.p = o;
        d.o = p;
        d.ans = -ans;
        test_data_vec.emplace_back(d);
        d.p = flipHorizontal(o);
        d.o = flipHorizontal(p);
        test_data_vec.emplace_back(d);
    }
    Data* train_data = &train_data_vec[0];
    Data* test_data = &test_data_vec[0];
    cout << "loaded data " << train_data_vec.size() << " " << test_data_vec.size() << endl;

    double grad_global = 0;
    double MSE_global = 0;
    double test_MAE = 0;
    double test_MSE = 0;
    size_t phase = 0;
    Pattern_Eval *pattern_arr_device;
    Adam* mobility_arr_device;
    Adam* stone_arr_device;
    double*grad_global_device;
    double* MSE_global_device;
    double* test_MAE_device;
    double* test_MSE_device;
    Data* train_data_device;
    Data* test_data_device;
    cudaMalloc((void**)&pattern_arr_device, sizeof(Pattern_Eval));
    cudaMalloc((void**)&mobility_arr_device, sizeof(Adam) * 36 * 36);
    cudaMalloc((void**)&stone_arr_device, sizeof(Adam) * 65 * 65);
    cudaMalloc((void**)&grad_global_device, sizeof(double));
    cudaMalloc((void**)&MSE_global_device, sizeof(double));
    cudaMalloc((void**)&test_MAE_device, sizeof(double));
    cudaMalloc((void**)&test_MSE_device, sizeof(double));
    cudaMalloc((void**)&train_data_device, train_data_vec.size() * sizeof(Data));
    cudaMalloc((void**)&test_data_device, test_data_vec.size() * sizeof(Data)); // Add allocation for test_data_device

    cudaMemcpy(train_data_device, train_data, train_data_vec.size() * sizeof(Data), cudaMemcpyHostToDevice);
    cudaMemcpy(test_data_device, test_data, test_data_vec.size() * sizeof(Data), cudaMemcpyHostToDevice); // Copy test data

    double MSE_BACK = 0;

    cout << "inited" << endl;
    cudaMemcpy(pattern_arr_device, &pattern_arr, sizeof(Pattern_Eval), cudaMemcpyHostToDevice);
    cudaMemcpy(mobility_arr_device, mobility_arr, sizeof(Adam) * 36 * 36, cudaMemcpyHostToDevice);
    cudaMemcpy(stone_arr_device, stone_arr, sizeof(Adam) * 65 * 65, cudaMemcpyHostToDevice);
    size_t dataSize = train_data_vec.size();
    int threadsPerBlock = 256;
    size_t blocks = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemset(pattern_arr_device, 0, sizeof(Pattern_Eval));
    cudaMemset(mobility_arr_device, 0, sizeof(Adam) * 36 * 36);
    cudaMemset(stone_arr_device, 0, sizeof(Adam) * 65 * 65);
    init_appear << <blocks, threadsPerBlock >> > (pattern_arr_device, mobility_arr_device, stone_arr_device, train_data_device, train_data_vec.size());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "init_appear launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
    phase = 0;
    auto start_time = high_resolution_clock::now();
    for (phase = 1; phase < ULONG_MAX; ++phase) { // Corrected phase variable type
        auto current_time = high_resolution_clock::now();
        auto elapsed = duration_cast<minutes>(current_time - start_time);
        if (elapsed.count() >= 5) { // 120分（= 2時間）
            cout << "\nTraining ended due to time limit.\n";
            break;
        }

        cudaMemcpy(grad_global_device, &grad_global, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(MSE_global_device, &MSE_global, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(test_MAE_device, &test_MAE, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(test_MSE_device, &test_MSE, sizeof(double), cudaMemcpyHostToDevice);
        cal_eval_diff << <blocks, threadsPerBlock >> > (pattern_arr_device, mobility_arr_device, stone_arr_device, train_data_device, grad_global_device, MSE_global_device,  dataSize);
        cudaDeviceSynchronize();

        cudaMemcpy(&pattern_arr, pattern_arr_device, sizeof(Pattern_Eval), cudaMemcpyDeviceToHost);
        cudaMemcpy(&grad_global, grad_global_device, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&MSE_global, MSE_global_device, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(mobility_arr, mobility_arr_device, sizeof(Adam) * 36 * 36, cudaMemcpyDeviceToHost);
        cudaMemcpy(stone_arr, stone_arr_device, sizeof(Adam) * 65 * 65, cudaMemcpyDeviceToHost);
        //cout << pattern_arr.diagonal8[0].t << " " << pattern_arr.diagonal8[0].w << " " << pattern_arr.diagonal8[0].delta << endl;
        for (auto& x : pattern_arr.diagonal8) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.diagonal7) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.diagonal6) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.diagonal5) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.edge_2x) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.h_v_2) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.h_v_3) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.h_v_4) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.corner_3x3) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.edge_x_side) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.edge_block) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.triangle) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : pattern_arr.corner_2x5) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : mobility_arr) {
           gd(x, alpha_t, phase);
        }
        for (auto& x : stone_arr) {
           gd(x, alpha_t, phase);
        }
        cudaMemcpy(pattern_arr_device, &pattern_arr, sizeof(Pattern_Eval), cudaMemcpyHostToDevice);
        cudaMemcpy(mobility_arr_device, mobility_arr, sizeof(Adam) * 36 * 36, cudaMemcpyHostToDevice);
        cudaMemcpy(stone_arr_device, stone_arr, sizeof(Adam) * 65 * 65, cudaMemcpyHostToDevice);
        if (test_data_vec.size() >0) cal_diff_test << < (test_data_vec.size() + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock >> > (pattern_arr_device, mobility_arr_device, stone_arr_device, test_data_device, test_MAE_device, test_MSE_device, test_data_vec.size());
        cudaDeviceSynchronize();
        cudaMemcpy(&test_MAE, test_MAE_device, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&test_MSE, test_MSE_device, sizeof(double), cudaMemcpyDeviceToHost);
        cout << "\rMAE : " << grad_global / train_data_vec.size() << " MSE : " << MSE_global / train_data_vec.size() << " TEST_MAE: " << test_MAE/test_data_vec.size() << " TEST_MSE: " << test_MSE/test_data_vec.size() << flush;
        //if (phase > 1 && MSE_global > MSE_BACK) {
        //    alpha_t *= 0.8;
        //    grad_global = 0;
        //    MSE_global = 0;
        //    MSE_BACK = MSE_global;
        //    continue;
        //}

        //if ((test_MSE > MSE_BACK && MSE_BACK > 0) || (grad_global /train_data_vec.size() + 1.0 < test_MAE / test_data_vec.size())) {
        //    cout << endl << "early stop" << endl;
        //    break;
        //}
        grad_global = 0;
        MSE_global = 0;
        MSE_BACK = test_MSE;
        test_MAE = 0;
        test_MSE = 0;

    }
    string txt = "out" + to_string(PHASE) + ".txt";
    ofstream ofs(txt);
    for (auto& x : pattern_arr.diagonal8) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.diagonal7) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.diagonal6) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.diagonal5) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.edge_2x) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.h_v_2) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.h_v_3) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.h_v_4) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.corner_3x3) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.edge_x_side) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.edge_block) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.triangle) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : pattern_arr.corner_2x5) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    for (auto& x : mobility_arr) {
        ofs << static_cast<int>(x.w) << "\n";
    }
    //for (auto& x : stone_arr) {
    //    ofs << static_cast<int>(x.w) << "\n";
    //}
}
