#ifndef _MODEL_SHAPE_H
#define _MODEL_SHAPE_H

#include <stdint.h>

// Parameter shapes of basic blocks
typedef struct BasicBlockShapes {
  int16_t bb_in_shape[3];
  int16_t bn_alpha_shape;

  int16_t skip_conv_bn_alpha_shape;
  int16_t skip_conv_bn_weight_shape[4]; // NHWC
  int16_t skip_conv_bn_stride[2];
  int16_t skip_conv_bn_padding[2];
  int16_t skip_conv_bn_io_shape[6];

  int16_t conv_bn_alpha_shape;
  int16_t conv_bn_weight_shape[4];      // NHWC
  int16_t conv_bn_stride[2];
  int16_t conv_bn_padding[2];
  int16_t conv_bn_io_shape[6];

  int16_t conv_weight_shape[4];         // NHWC
  int16_t conv_stride[2];
  int16_t conv_padding[2];
  int16_t conv_io_shape[6];

} BasicBlockShapes;

// Parameter shapes of other blocks (e.g., out of basic blocks)
typedef struct OtherShapes {
  int16_t bn_alpha_shape;
  int16_t fc_weight_shape[2];
  int16_t fc_bias_shape;
} OtherShapes;

const int16_t NUM_BASIC_BLOCKS = 5;
const int16_t NUM_CLASSES = 10;
const BasicBlockShapes bbShapes[NUM_BASIC_BLOCKS] = {
  // Layer 0
  { {32, 32, 80},
	80,                    // bn_alpha_beta
    0,  {0, 0, 0, 0},      // skip_conv_bn_alph_beta, skip_conv_bn_weight 
    {0, 0}, {0, 0},        // skip_conv_bn_stride, skip_conv_bn_padding
	{-1, -1, -1, -1, -1, -1}, // skip_conv_bn_io_shape
    80, {3, 3, 80, 80},    // conv_bn_alpha_beta, conv_bn_weight
    {1, 1}, {1, 1},        // conv_bn_stride, conv_bn_padding
	{32, 32, 80, 32, 32, 80}, // conv_bn_io_shape
    {3, 3, 80, 80},        // conv_weight
    {1, 1}, {1, 1},       // conv_stride, conv_padding
    {32, 32, 80, 32, 32, 80} // conv_io_shape
  },
  // Layer 1
  { {32, 32, 80},
	80,
    160, {1, 1, 160, 80},
    {2, 2}, {0, 0},
	{32, 32, 80, 16, 16, 160},
    160, {3, 3, 160, 80},
    {2, 2}, {1, 1},
	{32, 32, 80, 16, 16, 160},
    {3, 3, 160, 160},
    {1, 1}, {1, 1},
	{16, 16, 160, 16, 16, 160},
  },
  // Layer 2
  { {16, 16, 160},
	160,
    0,    {0, 0, 0, 0}, 
    {0, 0}, {0, 0},
	{-1, -1, -1, -1, -1, -1},
    160,  {3, 3, 160, 160},
    {1, 1}, {1, 1},
	{16, 16, 160, 16, 16, 160},
    {3, 3, 160, 160},
    {1, 1}, {1, 1},
	{16, 16, 160, 16, 16, 160},
  },
  // Layer 3
  { {16, 16, 160},
	160,
    320,  {1, 1, 320, 160},
    {2, 2}, {0, 0},
	{16, 16, 160, 8, 8, 320},
    320,  {3, 3, 320, 160},
    {2, 2}, {1, 1},
	{16, 16, 160, 8, 8, 320},
    {3, 3, 320, 320},
    {1, 1}, {1, 1},
	{8, 8, 320, 8, 8, 320}
  },
  // Layer 4
  { {8, 8, 320},
	320,
    0,    {0, 0, 0, 0}, 
    {0, 0}, {0, 0},
	{-1, -1, -1, -1, -1, -1},
    0,    {0, 0, 0, 0}, 
    {0, 0}, {0, 0},
	{-1, -1, -1, -1, -1, -1},
    {3, 3, 320, 320},
    {1, 1}, {1, 1},
	{8, 8, 320, 8, 8, 320},
  }
};
const OtherShapes otherShapes = {
//  bn    fc_weight            fc_bias
    320,  {NUM_CLASSES, 320},  NUM_CLASSES
};

#endif

