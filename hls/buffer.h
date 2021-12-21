#ifndef _BUFFER_H
#define _BUFFER_H

#include <ap_int.h>

#include "model_shape.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define AP_SIZE 16
#define GET_WEIGHT_ARRAY_LEN(x) ((x) / (AP_SIZE))


typedef ap_uint<AP_SIZE> WEIGHT_T;

// BatchNorm length
typedef struct BN_SIZE {
	int bn;
	int skip_conv_bn;
	int conv_bn;
} BN_SIZE;

typedef struct WEIGHT_SIZE {
	int skip_conv_bn;
	int conv_bn;
	int conv;
} WEIGHT_SIZE;

const BN_SIZE bnSize[NUM_BASIC_BLOCKS] = {
	{
			bbShapes[0].bn_alpha_shape,
			bbShapes[0].skip_conv_bn_alpha_shape,
			bbShapes[0].conv_bn_alpha_shape
	},
	{
			bbShapes[1].bn_alpha_shape,
			bbShapes[1].skip_conv_bn_alpha_shape,
			bbShapes[1].conv_bn_alpha_shape
	},
	{
			bbShapes[2].bn_alpha_shape,
			bbShapes[2].skip_conv_bn_alpha_shape,
			bbShapes[2].conv_bn_alpha_shape
	},
	{
			bbShapes[3].bn_alpha_shape,
			bbShapes[3].skip_conv_bn_alpha_shape,
			bbShapes[3].conv_bn_alpha_shape
	},
	{
			bbShapes[4].bn_alpha_shape,
			bbShapes[4].skip_conv_bn_alpha_shape,
			bbShapes[4].conv_bn_alpha_shape
	},
};


const WEIGHT_SIZE weightSize[NUM_BASIC_BLOCKS] = {
  {
    GET_WEIGHT_ARRAY_LEN(
      bbShapes[0].skip_conv_bn_weight_shape[0] * bbShapes[1].skip_conv_bn_weight_shape[1] *
      bbShapes[0].skip_conv_bn_weight_shape[2] * bbShapes[1].skip_conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[0].conv_bn_weight_shape[0] * bbShapes[0].conv_bn_weight_shape[1] *
      bbShapes[0].conv_bn_weight_shape[2] * bbShapes[0].conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[0].conv_weight_shape[0] * bbShapes[0].conv_weight_shape[1] *
      bbShapes[0].conv_weight_shape[2] * bbShapes[0].conv_weight_shape[3])
  },
  {
    GET_WEIGHT_ARRAY_LEN(
      bbShapes[1].skip_conv_bn_weight_shape[0] * bbShapes[1].skip_conv_bn_weight_shape[1] *
      bbShapes[1].skip_conv_bn_weight_shape[2] * bbShapes[1].skip_conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[1].conv_bn_weight_shape[0] * bbShapes[1].conv_bn_weight_shape[1] *
      bbShapes[1].conv_bn_weight_shape[2] * bbShapes[1].conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[1].conv_weight_shape[0] * bbShapes[1].conv_weight_shape[1] *
      bbShapes[1].conv_weight_shape[2] * bbShapes[1].conv_weight_shape[3])
  },
  {
    GET_WEIGHT_ARRAY_LEN(
      bbShapes[2].skip_conv_bn_weight_shape[0] * bbShapes[1].skip_conv_bn_weight_shape[1] *
      bbShapes[2].skip_conv_bn_weight_shape[2] * bbShapes[1].skip_conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[2].conv_bn_weight_shape[0] * bbShapes[2].conv_bn_weight_shape[1] *
      bbShapes[2].conv_bn_weight_shape[2] * bbShapes[2].conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[2].conv_weight_shape[0] * bbShapes[2].conv_weight_shape[1] *
      bbShapes[2].conv_weight_shape[2] * bbShapes[2].conv_weight_shape[3])
  },
  {
    GET_WEIGHT_ARRAY_LEN(
      bbShapes[3].skip_conv_bn_weight_shape[0] * bbShapes[3].skip_conv_bn_weight_shape[1] *
      bbShapes[3].skip_conv_bn_weight_shape[2] * bbShapes[3].skip_conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[3].conv_bn_weight_shape[0] * bbShapes[3].conv_bn_weight_shape[1] *
      bbShapes[3].conv_bn_weight_shape[2] * bbShapes[3].conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[3].conv_weight_shape[0] * bbShapes[3].conv_weight_shape[1] *
      bbShapes[3].conv_weight_shape[2] * bbShapes[3].conv_weight_shape[3])
  },
  {
    GET_WEIGHT_ARRAY_LEN(
      bbShapes[4].skip_conv_bn_weight_shape[0] * bbShapes[1].skip_conv_bn_weight_shape[1] *
      bbShapes[4].skip_conv_bn_weight_shape[2] * bbShapes[1].skip_conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[4].conv_bn_weight_shape[0] * bbShapes[4].conv_bn_weight_shape[1] *
      bbShapes[4].conv_bn_weight_shape[2] * bbShapes[4].conv_bn_weight_shape[3]),

    GET_WEIGHT_ARRAY_LEN(
      bbShapes[4].conv_weight_shape[0] * bbShapes[4].conv_weight_shape[1] *
      bbShapes[4].conv_weight_shape[2] * bbShapes[4].conv_weight_shape[3])
  },
};

#endif
