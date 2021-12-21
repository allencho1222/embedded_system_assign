#ifndef _RUN_H
#define _RUN_H

#include "buffer.h"
#include "model_shape.h"
#include "model_param.h"

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
  void* outer_fc_weight, void* outer_fc_bias);

#endif
