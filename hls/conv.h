#ifndef _CONV_H
#define _CONV_H

#include "buffer.h"

void conv(float* bb_bn_alpha_beta,
          float* bb_skip_conv_bn_alpha_beta,
          float* bb_conv_bn_alpha_beta,
          WEIGHT_T* bb_skip_conv_bn_weight,
          WEIGHT_T* bb_conv_bn_weight,
          WEIGHT_T* bb_conv_weight,
          int skip_conv_bn_stride_w, int skip_conv_bn_stride_h,
          int conv_bn_stride_w, int conv_bn_stride_h,
          int conv_stride_w, int conv_stride_h,
          int skip_conv_bn_padding_w, int skip_conv_bn_padding_h,
          int conv_bn_padding_w, int conv_bn_padding_h,
          int conv_pading_w, int conv_padding_h);

#endif

