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

BN_SIZE bnSize[NUM_BASIC_BLOCKS] = {
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

BN_SIZE bnSrcOffset[NUM_BASIC_BLOCKS] = {
{
  

WEIGHT_SIZE weightSize[NUM_BASIC_BLOCKS] = {
  {
    GET_ARRAY_LEN(
      bbShapes[0].skip_conv_bn_weight_shape[0] * bbShapes[1].skip_conv_bn_weight_shape[1] *
      bbShapes[0].skip_conv_bn_weight_shape[2] * bbShapes[1].skip_conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[0].conv_bn_weight_shape[0] * bbShapes[0].conv_bn_weight_shape[1] *
      bbShapes[0].conv_bn_weight_shape[2] * bbShapes[0].conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[0].conv_weight_shape[0] * bbShapes[0].conv_weight_shape[1] *
      bbShapes[0].conv_weight_shape[2] * bbShapes[0].conv_weight_shape[3])
  },
  {
    GET_ARRAY_LEN(
      bbShapes[1].skip_conv_bn_weight_shape[0] * bbShapes[1].skip_conv_bn_weight_shape[1] *
      bbShapes[1].skip_conv_bn_weight_shape[2] * bbShapes[1].skip_conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[1].conv_bn_weight_shape[0] * bbShapes[1].conv_bn_weight_shape[1] *
      bbShapes[1].conv_bn_weight_shape[2] * bbShapes[1].conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[1].conv_weight_shape[0] * bbShapes[1].conv_weight_shape[1] *
      bbShapes[1].conv_weight_shape[2] * bbShapes[1].conv_weight_shape[3])
  },
  {
    GET_ARRAY_LEN(
      bbShapes[2].skip_conv_bn_weight_shape[0] * bbShapes[1].skip_conv_bn_weight_shape[1] *
      bbShapes[2].skip_conv_bn_weight_shape[2] * bbShapes[1].skip_conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[2].conv_bn_weight_shape[0] * bbShapes[2].conv_bn_weight_shape[1] *
      bbShapes[2].conv_bn_weight_shape[2] * bbShapes[2].conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[2].conv_weight_shape[0] * bbShapes[2].conv_weight_shape[1] *
      bbShapes[2].conv_weight_shape[2] * bbShapes[2].conv_weight_shape[3])
  },
  {
    GET_ARRAY_LEN(
      bbShapes[3].skip_conv_bn_weight_shape[0] * bbShapes[3].skip_conv_bn_weight_shape[1] *
      bbShapes[3].skip_conv_bn_weight_shape[2] * bbShapes[3].skip_conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[3].conv_bn_weight_shape[0] * bbShapes[3].conv_bn_weight_shape[1] *
      bbShapes[3].conv_bn_weight_shape[2] * bbShapes[3].conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[3].conv_weight_shape[0] * bbShapes[3].conv_weight_shape[1] *
      bbShapes[3].conv_weight_shape[2] * bbShapes[3].conv_weight_shape[3])
  },
  {
    GET_ARRAY_LEN(
      bbShapes[4].skip_conv_bn_weight_shape[0] * bbShapes[1].skip_conv_bn_weight_shape[1] *
      bbShapes[4].skip_conv_bn_weight_shape[2] * bbShapes[1].skip_conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[4].conv_bn_weight_shape[0] * bbShapes[4].conv_bn_weight_shape[1] *
      bbShapes[4].conv_bn_weight_shape[2] * bbShapes[4].conv_bn_weight_shape[3]),

    GET_ARRAY_LEN(
      bbShapes[4].conv_weight_shape[0] * bbShapes[4].conv_weight_shape[1] *
      bbShapes[4].conv_weight_shape[2] * bbShapes[4].conv_weight_shape[3])
  },
};

// To calculate the maximum array size of bn parameters
#define BB_BN_ALPHA_BETA_LEN \
	MAX(bnSize[0].bn + bnSize[1].bn + bnSize[2].bn, \
		bnSize[3].bn + bnSize[4].bn)
#define BB_SKIP_CONV_BN_ALPHA_BETA_LEN \
	MAX(bnSize[0].skip_conv_bn + bnSize[1].skip_conv_bn + bnSize[2].skip_conv_bn, \
		bnSize[3].skip_conv_bn + bnSize[4].skip_conv_bn)
#define BB_CONV_BN_ALPHA_BETA_LEN \
	MAX(bnSize[0].conv_bn + bnSize[1].conv_bn + bnSize[2].conv_bn, \
		bnSize[3].conv_bn + bnSize[4].conv_bn)

// BatchNorm buffers
float bb_skip_conv_bn_alpha_beta[BB_SKIP_CONV_BN_ALPHA_BETA_LEN];
float bb_conv_bn_alpha_beta[BB_CONV_BN_ALPHA_BETA_LEN];
float bb_bn_alpha_beta[BB_BN_ALPHA_BETA_LEN];


// To calculate the maximum array size of conv weight
#define BB_SKIP_CONV_BN_WEIGHT_LEN \
	MAX(weightSize[0].skip_conv_bn + weightSize[1].skip_conv_bn +weightSize[2].skip_conv_bn, \
		weightSize[3].skip_conv_bn + weightSize[4].skip_conv_bn)
#define BB_CONV_BN_WEIGHT_LEN \
	MAX(weightSize[0].conv_bn + weightSize[1].conv_bn + weightSize[2].conv_bn, \
		weightSize[3].conv_bn + weightSize[4].conv_bn)
#define BB_CONV_WEIGHT_LEN \
	MAX(weightSize[0].conv + weightSize[1].conv + weightSize[2].conv, \
		weightSize[3].conv + weightSize[4].conv)

// Conv weight buffers
WEIGHT_T bb_skip_conv_bn_weight[BB_SKIP_CONV_BN_WEIGHT_LEN];
WEIGHT_T bb_conv_bn_weight[BB_CONV_BN_WEIGHT_LEN];
WEIGHT_T bb_conv_weight[BB_CONV_WEIGHT_LEN];

// OUTER
#define OUTER_ALPHA_BETA_LEN (otherShapes.bn_alpha_shape)
#define OUTER_FC_WEIGHT_LEN \
  GET_WEIGHT_ARRAY_LEN( \
    otherShapes.fc_weight_shape[0] * otherShapes.fc_weight_shape[1])
#define OUTER_FC_BIAS_LEN (otherShapes.fc_bias_shape)

float outer_bn_alpha_beta[OUTER_ALPHA_BETA_LEN];
WEIGHT_T outer_fc_weight[OUTER_FC_WEIGHT_LEN];
WEIGHT_T outer_fc_bias[OUTER_FC_BIAS_LEN];

float mm[320*3*3];

#endif

