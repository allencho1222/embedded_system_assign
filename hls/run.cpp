#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "run.h"
#include "conv.h"

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

  // To calculate the maximum array size of BN parameters
  const int BB_BN_ALPHA_BETA_LEN =
	MAX(bnSize[0].bn + bnSize[1].bn + bnSize[2].bn, \
      bnSize[3].bn + bnSize[4].bn);
  const int BB_SKIP_CONV_BN_ALPHA_BETA_LEN =
  	MAX(bnSize[0].skip_conv_bn + bnSize[1].skip_conv_bn + bnSize[2].skip_conv_bn, \
        bnSize[3].skip_conv_bn + bnSize[4].skip_conv_bn);
  const int BB_CONV_BN_ALPHA_BETA_LEN =
  	MAX(bnSize[0].conv_bn + bnSize[1].conv_bn + bnSize[2].conv_bn, \
        bnSize[3].conv_bn + bnSize[4].conv_bn);

  // BatchNorm buffers
  float bb_bn_alpha_beta[BB_BN_ALPHA_BETA_LEN];
  float bb_skip_conv_bn_alpha_beta[BB_SKIP_CONV_BN_ALPHA_BETA_LEN];
  float bb_conv_bn_alpha_beta[BB_CONV_BN_ALPHA_BETA_LEN];

  // To calculate the maximum array size of conv weight
  const int BB_SKIP_CONV_BN_WEIGHT_LEN =
  	MAX(weightSize[0].skip_conv_bn + weightSize[1].skip_conv_bn +weightSize[2].skip_conv_bn, \
  		  weightSize[3].skip_conv_bn + weightSize[4].skip_conv_bn);
  const int BB_CONV_BN_WEIGHT_LEN =
  	MAX(weightSize[0].conv_bn + weightSize[1].conv_bn + weightSize[2].conv_bn, \
  	    weightSize[3].conv_bn + weightSize[4].conv_bn);
  const int BB_CONV_WEIGHT_LEN =
  	MAX(weightSize[0].conv + weightSize[1].conv + weightSize[2].conv, \
  	    weightSize[3].conv + weightSize[4].conv);

  // Conv weight buffers
  WEIGHT_T bb_skip_conv_bn_weight[BB_SKIP_CONV_BN_WEIGHT_LEN];
  WEIGHT_T bb_conv_bn_weight[BB_CONV_BN_WEIGHT_LEN];
  WEIGHT_T bb_conv_weight[BB_CONV_WEIGHT_LEN];

  // Init weights for layer 0, 1, 2
  init_bb(bb_bn_alpha_beta,
          bb_skip_conv_bn_alpha_beta,
          bb_conv_bn_alpha_beta,
          bb_skip_conv_bn_weight,
          bb_conv_bn_weight,
          bb_conv_weight,
          (float*)bb_0_bn, (WEIGHT_T*)bb_0_weight,
          (float*)bb_1_bn, (WEIGHT_T*)bb_1_weight,
          (float*)bb_2_bn, (WEIGHT_T*)bb_2_weight);

  int bn_offset_0 = 0;
  int bn_offset_1 = 0;
  int bn_offset_2 = 0;

  int w_offset_0 = 0;
  int w_offset_1 = 0;
  int w_offset_2 = 0;
  // Run conv
  for (int i = 0; i < 3; ++i) {
	bn_offset_0 += ((i != 0) ? bnSize[i - 1].bn : 0);
	bn_offset_1 += ((i != 0) ? bnSize[i - 1].skip_conv_bn : 0);
	bn_offset_2 += ((i != 0) ? bnSize[i - 1].conv_bn : 0);
	w_offset_0 += ((i != 0) ? weightSize[i - 1].skip_conv_bn : 0);
	w_offset_1 += ((i != 0) ? weightSize[i - 1].conv_bn: 0);
	w_offset_2 += ((i != 0) ? weightSize[i - 1].conv : 0);
	conv(bb_bn_alpha_beta + bn_offset_0,
       bb_skip_conv_bn_alpha_beta + bn_offset_1,
       bb_conv_bn_alpha_beta + bn_offset_2,
       bb_skip_conv_bn_weight + w_offset_0,
       bb_conv_bn_weight + w_offset_1,
       bb_conv_weight + w_offset_2,
       bbShapes[i].skip_conv_bn_stride[0], bbShapes[i].skip_conv_bn_stride[1],
       bbShapes[i].conv_bn_stride[0], bbShapes[i].conv_bn_stride[1],
       bbShapes[i].conv_stride[0], bbShapes[i].conv_stride[1],
       bbShapes[i].skip_conv_bn_padding[0], bbShapes[i].skip_conv_bn_padding[i],
       bbShapes[i].conv_bn_stride[0], bbShapes[i].conv_bn_stride[1],
       bbShapes[i].conv_stride[0], bbShapes[i].conv_stride[1]);
  }
  // Init weights for layer 3, 4
  init_bb(bb_bn_alpha_beta,
          bb_skip_conv_bn_alpha_beta,
          bb_conv_bn_alpha_beta,
          bb_skip_conv_bn_weight,
          bb_conv_bn_weight,
          bb_conv_weight,
          (float*)bb_3_bn, (WEIGHT_T*)bb_3_weight,
          (float*)bb_4_bn, (WEIGHT_T*)bb_4_weight,
          nullptr, nullptr);
  // Run conv
  for (int i = 0; i < 2; ++i) {
  	bn_offset_0 += ((i != 0) ? bnSize[i - 1 + 3].bn : 0);
  	bn_offset_1 += ((i != 0) ? bnSize[i - 1 + 3].skip_conv_bn : 0);
  	bn_offset_2 += ((i != 0) ? bnSize[i - 1 + 3].conv_bn : 0);
  	w_offset_0 += ((i != 0) ? weightSize[i - 1 + 3].skip_conv_bn : 0);
  	w_offset_1 += ((i != 0) ? weightSize[i - 1 + 3].conv_bn: 0);
  	w_offset_2 += ((i != 0) ? weightSize[i - 1 + 3].conv : 0);
  	conv(bb_bn_alpha_beta + bn_offset_0,
         bb_skip_conv_bn_alpha_beta + bn_offset_1,
         bb_conv_bn_alpha_beta + bn_offset_2,
         bb_skip_conv_bn_weight + w_offset_0,
         bb_conv_bn_weight + w_offset_1,
         bb_conv_weight + w_offset_2,
		 bbShapes[i + 3].skip_conv_bn_stride[0], bbShapes[i + 3].skip_conv_bn_stride[1],
		 bbShapes[i + 3].conv_bn_stride[0], bbShapes[i + 3].conv_bn_stride[1],
		 bbShapes[i + 3].conv_stride[0], bbShapes[i + 3].conv_stride[1],
		 bbShapes[i + 3].skip_conv_bn_padding[0], bbShapes[i + 3].skip_conv_bn_padding[1],
		 bbShapes[i + 3].conv_bn_stride[0], bbShapes[i + 3].conv_bn_stride[1],
		 bbShapes[i + 3].conv_stride[0], bbShapes[i + 3].conv_stride[1]);
    }
  // TODO: need to implement outer bn and fc
  //init_outer(outer_bn, outer_fc_weight, outer_fc_bias);
}

void init_bb(
  float* bb_bn_alpha_beta,
  float* bb_skip_conv_bn_alpha_beta,
  float* bb_conv_bn_alpha_beta,
  WEIGHT_T* bb_skip_conv_bn_weight,
  WEIGHT_T* bb_conv_bn_weight,
  WEIGHT_T* bb_conv_weight,
  float* bb_0_bn, WEIGHT_T* bb_0_weight,
  float* bb_1_bn, WEIGHT_T* bb_1_weight,
  float* bb_2_bn, WEIGHT_T* bb_2_weight) {

  int src_offset = 0;

  int bn_dest_offset_0 = 0;
  int bn_dest_offset_1 = 0;
  int bn_dest_offset_2 = 0;

  int w_dest_offset_0 = 0;
  int w_dest_offset_1 = 0;
  int w_dest_offset_2 = 0;

  // Layer 0
   src_offset = 0;
   bn_dest_offset_0 += 0;
   memcpy(bb_bn_alpha_beta + bn_dest_offset_0, bb_0_bn + src_offset, sizeof(float) * bnSize[0].bn);
   src_offset += bnSize[0].bn;
   bn_dest_offset_1 += 0;
   memcpy(bb_skip_conv_bn_alpha_beta + bn_dest_offset_1, bb_0_bn + src_offset, sizeof(float) * bnSize[0].skip_conv_bn);
   src_offset += bnSize[0].skip_conv_bn;
   bn_dest_offset_2 += 0;
   memcpy(bb_conv_bn_alpha_beta + bn_dest_offset_2, bb_0_bn + src_offset, sizeof(float) * bnSize[0].conv_bn);

   src_offset = 0;
   w_dest_offset_0 += 0;
   memcpy(bb_skip_conv_bn_weight + w_dest_offset_0, bb_0_weight + src_offset, sizeof(WEIGHT_T) * weightSize[0].skip_conv_bn);
   src_offset += weightSize[0].skip_conv_bn;
   w_dest_offset_1 += 0;
   memcpy(bb_conv_bn_weight + w_dest_offset_1, bb_0_weight + src_offset, sizeof(WEIGHT_T) * weightSize[0].conv_bn);
   src_offset += weightSize[0].conv_bn;
   w_dest_offset_2 += 0;
   memcpy(bb_conv_weight + w_dest_offset_2, bb_0_weight + src_offset, sizeof(WEIGHT_T) * weightSize[0].conv);

   // Layer 1
   src_offset = 0;
   bn_dest_offset_0 += bnSize[0].bn;
   memcpy(bb_bn_alpha_beta + bn_dest_offset_0, bb_1_bn + src_offset, sizeof(float) * bnSize[1].bn);
   src_offset += bnSize[1].bn;
   bn_dest_offset_1 += bnSize[0].skip_conv_bn;
   memcpy(bb_skip_conv_bn_alpha_beta + bn_dest_offset_1, bb_1_bn + src_offset, sizeof(float) * bnSize[1].skip_conv_bn);
   src_offset += bnSize[1].skip_conv_bn;
   bn_dest_offset_2 += bnSize[0].conv_bn;
   memcpy(bb_conv_bn_alpha_beta + bn_dest_offset_2, bb_1_bn + src_offset, sizeof(float) * bnSize[1].conv_bn);

   src_offset = 0;
   w_dest_offset_0 += weightSize[0].skip_conv_bn;
   memcpy(bb_skip_conv_bn_weight + w_dest_offset_0, bb_1_weight + src_offset, sizeof(WEIGHT_T) * weightSize[1].skip_conv_bn);
   src_offset += weightSize[1].skip_conv_bn;
   w_dest_offset_1 += weightSize[0].conv_bn;
   memcpy(bb_conv_bn_weight + w_dest_offset_1, bb_1_weight + src_offset, sizeof(WEIGHT_T) * weightSize[1].conv_bn);
   src_offset += weightSize[1].conv_bn;
   w_dest_offset_2 += weightSize[0].conv;
   memcpy(bb_conv_weight + w_dest_offset_2, bb_1_weight + src_offset, sizeof(WEIGHT_T) * weightSize[1].conv);

   if (bb_2_bn != nullptr && bb_2_weight != nullptr) {
     // Layer 2
     src_offset = 0;
     bn_dest_offset_0 += bnSize[1].bn;
     memcpy(bb_bn_alpha_beta + bn_dest_offset_0, bb_2_bn + src_offset, sizeof(float) * bnSize[2].bn);
     src_offset += bnSize[2].bn;
     bn_dest_offset_1 += bnSize[1].skip_conv_bn;
     memcpy(bb_skip_conv_bn_alpha_beta + bn_dest_offset_1, bb_2_bn + src_offset, sizeof(float) * bnSize[2].skip_conv_bn);
     src_offset += bnSize[2].skip_conv_bn;
     bn_dest_offset_2 += bnSize[1].conv_bn;
     memcpy(bb_conv_bn_alpha_beta + bn_dest_offset_2, bb_2_bn + src_offset, sizeof(float) * bnSize[2].conv_bn);

     src_offset = 0;
     w_dest_offset_0 += weightSize[1].skip_conv_bn;
     memcpy(bb_skip_conv_bn_weight + w_dest_offset_0, bb_2_weight + src_offset, sizeof(WEIGHT_T) * weightSize[2].skip_conv_bn);
     src_offset += weightSize[2].skip_conv_bn;
     w_dest_offset_1 += weightSize[1].conv_bn;
     memcpy(bb_conv_bn_weight + w_dest_offset_1, bb_2_weight + src_offset, sizeof(WEIGHT_T) * weightSize[2].conv_bn);
     src_offset += weightSize[2].conv_bn;
     w_dest_offset_2 += weightSize[1].conv;
     memcpy(bb_conv_weight + w_dest_offset_2, bb_2_weight + src_offset, sizeof(WEIGHT_T) * weightSize[2].conv);
   }
}





