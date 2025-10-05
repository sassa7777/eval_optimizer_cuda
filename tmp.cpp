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
