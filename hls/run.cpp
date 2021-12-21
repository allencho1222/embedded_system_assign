void run(
  // Input to basic block 0
  void* input,

  // Paramters for each basic block
  // For each basic block, parameters are listed by:
  // bn_alpha_beta, skip_conv_bn_alpha_beta, conv_bn_alpha_beta,
  // skip_conv_bn_weight, conv_bn_weight, conv_weight
  void* bb_0_bn, void* bb_0_weight,
  void* bb_1_bn, void* bb_1_weight,
  void* bb_2_bn, void* bb_2_weight,
  void* bb_3_bn, void* bb_3_weight,
  void* bb_4_bn, void* bb_4_weight,

  // Parameters for outer block (BN and FC)
  // Parameters are listed by:
  // bn_alpha_beta, fc_weight, fc_bias
  void* outer_bn,
  void* outer_fc_weight, void* outer_fc_bias) {

  // M_AXI port declaration
  #pragma HLS INTERFACE m_axi port = input depth = 4096 offset = slave bundle = a_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_0_bn depth = 4096 offset = slave bundle = b_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_0_weight depth = 4096 offset = slave bundle = c_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_1_bn depth = 4096 offset = slave bundle = d_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_1_weight depth = 4096 offset = slave bundle = e_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_2_bn depth = 4096 offset = slave bundle = f_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_2_weight depth = 4096 offset = slave bundle = g_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_3_bn depth = 4096 offset = slave bundle = h_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_3_weight depth = 4096 offset = slave bundle = i_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_4_bn depth = 4096 offset = slave bundle = j_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = bb_4_weight depth = 4096 offset = slave bundle = k_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = outer_bn depth = 4096 offset = slave bundle = l_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = outer_fc_weight depth = 4096 offset = slave bundle = m_port max_read_burst_length = 64
  #pragma HLS INTERFACE m_axi port = outer_fc_bias depth = 4096 offset = slave bundle = n_port max_read_burst_length = 64

  // S_AXI port declaration
  #pragma HLS INTERFACE s_axilite port = input bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_0_bn bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_0_weight bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_1_bn bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_1_weight bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_2_bn bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_2_weight bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_3_bn bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_3_weight bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_4_bn bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = bb_4_weight bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = outer_bn bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = outer_fc_weight bundle = CONTROL_BUS
  #pragma HLS INTERFACE s_axilite port = outer_fc_bias bundle = CONTROL_BUS

  #pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  init_bb_0_1_2(bb_0_bn, bb_0_weight,
                bb_1_bn, bb_1_weight,
                bb_2_bn, bb_2_weight);
  init_bb(bb_0_bn, bb_0_weight,
          bb_1_bn, bb_1_weight,
          bb_2_bn, bb_2_weight,
          0, 2);
  init_bb(bb_3_bn, bb_3_weight,
          bb_4_bn, bb_4_weight,
          nullptr, nullptr
          3, 4);
  //init_outer(outer_bn, outer_fc_weight, outer_fc_bias);
}

void init_bb(
  float* bb_0_bn, WEIGHT_T* bb_0_weight,
  float* bb_1_bn, WEIGHT_T* bb_1_weight,
  float* bb_2_bn, WEIGHT_T* bb_2_weight,
  int start, int end) {

  int src_offset = 0;

  int bn_dest_offset_0 = 0;
  int bn_dest_offset_1 = 0;
  int bn_dest_offset_2 = 0;

  int w_dest_offset_0 = 0;
  int w_dest_offset_1 = 0;
  int w_dest_offset_2 = 0;

  float* bn[3] = {bb_0_bn, bb_1_bn, bb_2_bn};
  float* w[3] = {bb_0_weight, bb_1_weight, bb_2_weight};

  for (int i = start; i <= end; ++i) {
    src_offset = 0;
    bn_dest_offset_0 += ((i - 1) < 0) ? 0: bnSize[i - 1].bn;
    memcpy(bb_bn_alpha_beta + bn_dest_offset_0, bn[i] + src_offset, sizeof(float) * bnSize[i].bn);
    src_offset += bnSize[i].bn;
    bn_dest_offset_1 += ((i - 1) < 0) ? 0 : bnSize[i - 1].skip_conv_bn;
    memcpy(bb_skip_conv_bn_alpha_beta + bn_dest_offset_1, bn[i] + src_offset, sizeof(float) * bnSize[i].skip_conv_bn);
    src_offset += bnSize[i].skip_conv_bn;
    bn_dest_offset_2 += ((i - 1) < 0) ? 0 : bnSize[i - 1].conv_bn;
    memcpy(bb_conv_bn_alpha_beta + bn_dest_offset_2, bn[i] + src_offset, sizeof(float) * bnSize[i].conv_bn);


    src_offset = 0;
    w_dest_offset_0 += ((i - 1) < 0) ? 0: weightSize[i - 1].skip_conv_bn;
    memcpy(bb_skip_conv_bn_weight + w_dest_offset_0, w[i] + src_offset, sizeof(float) * weightSize[i].skip_conv_bn);
    src_offset += weightSize[i].skip_conv_bn;
    w_dest_offset_1 += ((i - 1) < 0) ? 0 : weightSize[i - 1].conv_bn;
    memcpy(bb_conv_bn_weight + w_dest_offset_1, w[i] + src_offset, sizeof(float) * weightSize[i].conv_bn);
    src_offset += bnSize[i].conv_bn;
    w_dest_offset_2 += ((i - 1) < 0) ? 0 : bnSize[i - 1].conv;
    memcpy(bb_conv_weight + w_dest_offset_2, w[i] + src_offset, sizeof(float) * weightSize[i].conv);
  }
}




