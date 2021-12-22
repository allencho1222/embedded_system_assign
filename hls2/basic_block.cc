#include "conv.h"

#define B1_IH 32
#define B1_IW 32
#define B1_IC 80
#define B1_OH1 32
#define B1_OW1 32
#define B1_OC1 80
#define B1_OH2 32
#define B1_OW2 32
#define B1_OC2 80
#define B1_SCALE
#define B1_STRIDE1 1
#define B1_STRIDE2 1

#define B2_IH 32
#define B2_IW 32
#define B2_IC 80
#define B2_OH1 16
#define B2_OW1 16
#define B2_OC1 160
#define B2_OH2 16
#define B2_OW2 16
#define B2_OC2 160
#define B2_SCALE
#define B2_STRIDE1 2
#define B2_STRIDE2 1

#define B3_IH 16
#define B3_IW 16
#define B3_IC 160
#define B3_OH1 16
#define B3_OW1 16
#define B3_OC1 160
#define B3_OH2 16
#define B3_OW2 16
#define B3_OC2 160
#define B3_SCALE
#define B3_STRIDE1 1
#define B3_STRIDE2 1

#define B4_IH 16
#define B4_IW 16
#define B4_IC 160
#define B4_OH1 8
#define B4_OW1 8
#define B4_OC1 320
#define B4_OH2 8
#define B4_OW2 8
#define B4_OC2 320
#define B4_SCALE
#define B4_STRIDE1 2
#define B4_STRIDE2 1

#define B5_IH 8
#define B5_IW 8
#define B5_IC 320
#define B5_OH 8
#define B5_OW 8
#define B5_OC 320
#define B5_STRIDE 1
#define IMAGES 10

void conv_b1(
    WEIGHT_T *bb_conv_bn_weight,
    WEIGHT_T *bb_conv_weight,
    WEIGHT_T *bb_skip_conv_bn_weight,
    float *bb_bn_alpha,
    float *bb_bn_beta,
    float *bb_skip_conv_bn_alpha,
    float *bb_skip_conv_bn_beta,
    float *bb_conv_bn_alpha,
    float *bb_conv_bn_beta,
    float *input_activation, float *output_activatoin)
{

  int temp_activation[B1_IH * B1_IW * B1_IC];
  int temp_output1_acgivation[B1_IH * B1_IW * B1_IC];
  float output_tile_oc[B1_OC1] = {0};
bn_relu0:
  for (int hw = 0; hw < B1_IH * B1_IW; hw++)
  {
    for (int ic = 0; ic < B1_IC; ic++)
    {
      temp_activation[hw * B1_IC + ic] = RELU(input_activation[hw * B1_IC + ic] * bb_bn_alpha[ic] + bb_bn_beta[ic]);
    }
  }
  for (int oh = 0; oh < B1_OH1; oh++)
  {
    for (int ow = 0; ow < B1_OW1; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * B1_STRIDE1) + kh - ALL_PADDING;
          int iw = (ow * B1_STRIDE1) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B1_IH && iw < B1_IW)
          {
          conv1:
            for (int oc = 0; oc < B1_OC1; oc++)
            {
              for (int ic = 0; ic < B1_IC; ic++)
              {
                if (bb_conv_bn_weight[(kh * OTHER_KERNEL_SIZE * B1_OC1 * B1_IC / AP_SIZE) + (kw * B1_OC1 * B1_IC / AP_SIZE) + (oc * B1_IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_activation[ih * B1_IW * B1_IC + iw * B1_IC + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_activation[ih * B1_IW * B1_IC + iw * B1_IC + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
    bn_relu1:
      for (int oc = 0; oc < B1_OC1; oc++)
      {
        temp_output1_acgivation[oh * B1_OW1 * B1_OC1 + ow * B1_OC1 + oc] = RELU(output_tile_oc[oc] * bb_conv_bn_alpha[oc] + bb_conv_bn_beta[oc]);
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
  for (int oh = 0; oh < B1_OH2; oh++)
  {
    for (int ow = 0; ow < B1_OW2; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * B1_STRIDE2) + kh - ALL_PADDING;
          int iw = (ow * B1_STRIDE2) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B1_OH1 && iw < B1_OW1)
          {
          conv2:
            for (int oc = 0; oc < B1_OC2; oc++)
            {
              for (int ic = 0; ic < B1_OC1; ic++)
              {
                if (bb_conv_weight[(kh * OTHER_KERNEL_SIZE * B1_OC2 * B1_OC1 / AP_SIZE) + (kw * B1_OC2 * B1_OC1 / AP_SIZE) + (oc * B1_OC1 / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_output1_acgivation[ih * B1_OW1 * B1_OC1 + iw * B1_OC1 + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_output1_acgivation[ih * B1_OW1 * B1_OC1 + iw * B1_OC1 + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
      for (int oc = 0; oc < B1_OC2; oc++)
      {
        output_activatoin[oh * B1_OW2 * B1_OC2 + ow * B1_OC2 + oc] = output_tile_oc[oc] * B1_SCALE + input_activation[oh * B1_OW2 * B1_OC2 + ow * B1_OC2 + oc];
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
}

void conv_b2(
    WEIGHT_T *bb_conv_bn_weight_dram,
    WEIGHT_T *bb_conv_weight_dram,
    WEIGHT_T *bb_skip_conv_bn_weight,
    float scale2,
    float *bb_bn_alpha,
    float *bb_bn_beta,
    float *bb_skip_conv_bn_alpha,
    float *bb_skip_conv_bn_beta,
    float *bb_conv_bn_alpha,
    float *bb_conv_bn_beta,
    float *input_activation, float *output_activatoin)
{

  int temp_activation[B2_IH * B2_IW * B2_IC];
  int temp_output1_acgivation[B2_IH * B2_IW * B2_IC];
  float output_tile_oc[B2_OC2] = {0};
  WEIGHT_T weight_buff[OTHER_KERNEL_SIZE * OTHER_KERNEL_SIZE * B2_OC2 * B2_OC1 / AP_SIZE];
  memcpy(&weight_buff[0], 
    const_cast<WEIGHT_T*>(&bb_conv_bn_weight_dram[0]),
    sizeof(WEIGHT_T) * OTHER_KERNEL_SIZE * OTHER_KERNEL_SIZE * B2_OC1 * B2_IC / AP_SIZE);

bn_relu0:
  for (int hw = 0; hw < B2_IH * B2_IW; hw++)
  {
    for (int ic = 0; ic < B2_IC; ic++)
    {
      temp_activation[hw * B2_IC + ic] = RELU(input_activation[hw * B2_IC + ic] * bb_bn_alpha[ic] + bb_bn_beta[ic]);
    }
  }

  for (int oh = 0; oh < B2_OH1; oh++)
  {
    for (int ow = 0; ow < B2_OW1; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * B2_STRIDE1) + kh - ALL_PADDING;
          int iw = (ow * B2_STRIDE1) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B2_IH && iw < B2_IW)
          {
          conv1:
            for (int oc = 0; oc < B2_OC1; oc++)
            {
              for (int ic = 0; ic < B2_IC; ic++)
              {
                if (weight_buff[(kh * OTHER_KERNEL_SIZE * B2_OC1 * B2_IC / AP_SIZE) + (kw * B2_OC1 * B2_IC / AP_SIZE) + (oc * B2_IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_activation[ih * B2_IW * B2_IC + iw * B2_IC + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_activation[ih * B2_IW * B2_IC + iw * B2_IC + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
    bn_relu1:
      for (int oc = 0; oc < B2_OC1; oc++)
      {
        temp_output1_acgivation[oh * B2_OW1 * B2_OC1 + ow * B2_OC1 + oc] = RELU(output_tile_oc[oc] * bb_conv_bn_alpha[oc] + bb_conv_bn_beta[oc]);
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
  for (int oh = 0; oh < B2_OH2; oh++)
  {
    for (int ow = 0; ow < B2_OW2; ow++)
    {
      for (int kh = 0; kh < SKIP_CONV_BN_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < SKIP_CONV_BN_KERNEL_SIZE; kw++)
        {
          int ih = (oh * SKIP_CONV_BN_STRIDE) + kh - SKIP_CONV_BN_PADDING;
          int iw = (ow * SKIP_CONV_BN_STRIDE) + kw - SKIP_CONV_BN_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B2_IH && iw < B2_IW)
          {
          conv_skip:
            for (int oc = 0; oc < B2_OC2; oc++)
            {
              for (int ic = 0; ic < B2_IC; ic++)
              {
                if (bb_skip_conv_bn_weight[(kh * SKIP_CONV_BN_KERNEL_SIZE * B2_OC2 * B2_IC / AP_SIZE) + (kw * B2_OC2 * B2_IC / AP_SIZE) + (oc * B2_IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_activation[ih * B2_IW * B2_IC + iw * B2_IC + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_activation[ih * B2_IW * B2_IC + iw * B2_IC + ic];
                }
              } // loop for IC
            }   // loop for OW
          }
        } // loop for KW
      }   // loop  for KH
    bn_skip:
      for (int oc = 0; oc < B2_OC2; oc++)
      { // Skip conv residual
        output_activatoin[oh * B2_OW2 * B2_OC2 + ow * B2_OC2 + oc] = output_tile_oc[oc] * bb_skip_conv_bn_alpha[oc] + bb_skip_conv_bn_beta[oc];
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
  memcpy(&weight_buff[0], 
    const_cast<WEIGHT_T*>(&bb_conv_weight_dram[0]),
    sizeof(WEIGHT_T) * OTHER_KERNEL_SIZE * OTHER_KERNEL_SIZE * B2_OC2 * B2_OC1 / AP_SIZE);
  for (int oh = 0; oh < B2_OH2; oh++)
  {
    for (int ow = 0; ow < B2_OW2; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * B2_STRIDE2) + kh - ALL_PADDING;
          int iw = (ow * B2_STRIDE2) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B2_OH1 && iw < B2_OW1)
          {
          conv2:
            for (int oc = 0; oc < B2_OC2; oc++)
            {
              for (int ic = 0; ic < B2_OC1; ic++)
              {
                if (weight_buff[(kh * OTHER_KERNEL_SIZE * B2_OC2 * B2_OC1 / AP_SIZE) + (kw * B2_OC2 * B2_OC1 / AP_SIZE) + (oc * B2_OC1 / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_output1_acgivation[ih * B2_OW1 * B2_OC1 + iw * B2_OC1 + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_output1_acgivation[ih * B2_OW1 * B2_OC1 + iw * B2_OC1 + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
      for (int oc = 0; oc < B2_OC2; oc++)
      {
        output_activatoin[oh * B2_OW2 * B2_OC2 + ow * B2_OC2 + oc] += output_tile_oc[oc] * B2_SCALE;
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
}

void conv_b3(
    WEIGHT_T *bb_conv_bn_weight_dram,
    WEIGHT_T *bb_conv_weight_dram,
    WEIGHT_T *bb_skip_conv_bn_weight,
    float *bb_bn_alpha,
    float *bb_bn_beta,
    float *bb_skip_conv_bn_alpha,
    float *bb_skip_conv_bn_beta,
    float *bb_conv_bn_alpha,
    float *bb_conv_bn_beta,
    float *input_activation, float *output_activatoin)
{

  int temp_activation[B3_IH * B3_IW * B3_IC];
  int temp_output1_acgivation[B3_IH * B3_IW * B3_IC];
  float output_tile_oc[B3_OC1] = {0};
  WEIGHT_T weight_buff[OTHER_KERNEL_SIZE * OTHER_KERNEL_SIZE * B3_OC2 * B3_OC1 / AP_SIZE];
  memcpy(&weight_buff[0], 
    const_cast<WEIGHT_T*>(&bb_conv_bn_weight_dram[0]),
    sizeof(WEIGHT_T) * OTHER_KERNEL_SIZE * OTHER_KERNEL_SIZE * B3_OC1 * B3_IC / AP_SIZE);
bn_relu0:
  for (int hw = 0; hw < B3_IH * B3_IW; hw++)
  {
    for (int ic = 0; ic < B3_IC; ic++)
    {
      temp_activation[hw * B3_IC + ic] = RELU(input_activation[hw * B3_IC + ic] * bb_bn_alpha[ic] + bb_bn_beta[ic]);
    }
  }
  for (int oh = 0; oh < B3_OH1; oh++)
  {
    for (int ow = 0; ow < B3_OW1; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * B3_STRIDE1) + kh - ALL_PADDING;
          int iw = (ow * B3_STRIDE1) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B3_IH && iw < B3_IW)
          {
          conv1:
            for (int oc = 0; oc < B3_OC1; oc++)
            {
              for (int ic = 0; ic < B3_IC; ic++)
              {
                if (bb_conv_bn_weight[(kh * OTHER_KERNEL_SIZE * B3_OC1 * B3_IC / AP_SIZE) 
                  + (kw * B3_OC1 * B3_IC / AP_SIZE) + (oc * B3_IC / AP_SIZE) 
                  + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_activation[ih * B3_IW * B3_IC + iw * B3_IC + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_activation[ih * B3_IW * B3_IC + iw * B3_IC + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
    bn_relu1:
      for (int oc = 0; oc < B3_OC1; oc++)
      {
        temp_output1_acgivation[oh * B3_OW1 * B3_OC1 + ow * B3_OC1 + oc] 
          = RELU(output_tile_oc[oc] * bb_conv_bn_alpha[oc] + bb_conv_bn_beta[oc]);
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
  memcpy(&weight_buff[0], 
  const_cast<WEIGHT_T*>(&bb_conv_weight_dram[0]),
    sizeof(WEIGHT_T) * OTHER_KERNEL_SIZE * OTHER_KERNEL_SIZE * B3_OC2 * B3_OC1 / AP_SIZE);
  for (int oh = 0; oh < B3_OH2; oh++)
  {
    for (int ow = 0; ow < B3_OW2; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * B3_STRIDE2) + kh - ALL_PADDING;
          int iw = (ow * B3_STRIDE2) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B3_OH1 && iw < B3_OW1)
          {
          conv2:
            for (int oc = 0; oc < B3_OC2; oc++)
            {
              for (int ic = 0; ic < B3_OC1; ic++)
              {
                if (bb_conv_weight[(kh * OTHER_KERNEL_SIZE * B3_OC2 * B3_OC1 / AP_SIZE) 
                  + (kw * B3_OC2 * B3_OC1 / AP_SIZE) + (oc * B3_OC1 / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_output1_acgivation[ih * B3_OW1 * B3_OC1 + iw * B3_OC1 + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_output1_acgivation[ih * B3_OW1 * B3_OC1 + iw * B3_OC1 + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
      for (int oc = 0; oc < B3_OC2; oc++)
      {
        output_activatoin[oh * B3_OW2 * B3_OC2 + ow * B3_OC2 + oc] 
          = output_tile_oc[oc] * B3_SCALE + input_activation[oh * B3_OW2 * B3_OC2 + ow * B3_OC2 + oc];
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
}

void conv_b4(
    WEIGHT_T *bb_conv_bn_weight_dram,
    WEIGHT_T *bb_conv_weight_dram,
    WEIGHT_T *bb_skip_conv_bn_weight,
    float scale2,
    float *bb_bn_alpha,
    float *bb_bn_beta,
    float *bb_skip_conv_bn_alpha,
    float *bb_skip_conv_bn_beta,
    float *bb_conv_bn_alpha,
    float *bb_conv_bn_beta,
    float *input_activation, float *output_activatoin)
{

  int temp_activation[B4_IH * B4_IW * B4_IC];
  int temp_output1_acgivation[B4_IH * B4_IW * B4_IC];
  float output_tile_oc[B4_OC2] = {0};
  WEIGHT_T weight_buff[OTHER_KERNEL_SIZE * OTHER_KERNEL_SIZE * B4_OC2 * B4_OC1 / AP_SIZE];
  memcpy(&weight_buff[0], 
    const_cast<WEIGHT_T*>(&bb_conv_bn_weight_dram[0]),
    sizeof(WEIGHT_T) * OTHER_KERNEL_SIZE * OTHER_KERNEL_SIZE * B4_OC1 * B4_IC / AP_SIZE);
bn_relu0:
  for (int hw = 0; hw < B4_IH * B4_IW; hw++)
  {
    for (int ic = 0; ic < B4_IC; ic++)
    {
      temp_activation[hw * B4_IC + ic] = RELU(input_activation[hw * B4_IC + ic] * bb_bn_alpha[ic] + bb_bn_beta[ic]);
    }
  }

  for (int oh = 0; oh < B4_OH1; oh++)
  {
    for (int ow = 0; ow < B4_OW1; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * B4_STRIDE1) + kh - ALL_PADDING;
          int iw = (ow * B4_STRIDE1) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B4_IH && iw < B4_IW)
          {
          conv1:
            for (int oc = 0; oc < B4_OC1; oc++)
            {
              for (int ic = 0; ic < B4_IC; ic++)
              {
                if (bb_conv_bn_weight[(kh * OTHER_KERNEL_SIZE * B4_OC1 * B4_IC / AP_SIZE) + (kw * B4_OC1 * B4_IC / AP_SIZE) + (oc * B4_IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_activation[ih * B4_IW * B4_IC + iw * B4_IC + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_activation[ih * B4_IW * B4_IC + iw * B4_IC + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
    bn_relu1:
      for (int oc = 0; oc < B4_OC1; oc++)
      {
        temp_output1_acgivation[oh * B4_OW1 * B4_OC1 + ow * B4_OC1 + oc] = RELU(output_tile_oc[oc] * bb_conv_bn_alpha[oc] + bb_conv_bn_beta[oc]);
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
  for (int oh = 0; oh < B4_OH2; oh++)
  {
    for (int ow = 0; ow < B4_OW2; ow++)
    {
      for (int kh = 0; kh < SKIP_CONV_BN_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < SKIP_CONV_BN_KERNEL_SIZE; kw++)
        {
          int ih = (oh * SKIP_CONV_BN_STRIDE) + kh - SKIP_CONV_BN_PADDING;
          int iw = (ow * SKIP_CONV_BN_STRIDE) + kw - SKIP_CONV_BN_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B4_IH && iw < B4_IW)
          {
          conv_skip:
            for (int oc = 0; oc < B4_OC2; oc++)
            {
              for (int ic = 0; ic < B4_IC; ic++)
              {
                if (bb_skip_conv_bn_weight[(kh * SKIP_CONV_BN_KERNEL_SIZE * B4_OC2 * B4_IC / AP_SIZE) + (kw * B4_OC2 * B4_IC / AP_SIZE) + (oc * B4_IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_activation[ih * B4_IW * B4_IC + iw * B4_IC + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_activation[ih * B4_IW * B4_IC + iw * B4_IC + ic];
                }
              } // loop for IC
            }   // loop for OW
          }
        } // loop for KW
      }   // loop  for KH
    bn_skip:
      for (int oc = 0; oc < B4_OC2; oc++)
      { // Skip conv residual
        output_activatoin[oh * B4_OW2 * B4_OC2 + ow * B4_OC2 + oc] = output_tile_oc[oc] * bb_skip_conv_bn_alpha[oc] + bb_skip_conv_bn_beta[oc];
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
  memcpy(&weight_buff[0], 
  const_cast<WEIGHT_T*>(&bb_conv_weight_dram[0]),
    sizeof(WEIGHT_T) * OTHER_KERNEL_SIZE * OTHER_KERNEL_SIZE * B4_OC2 * B4_OC1 / AP_SIZE);
  for (int oh = 0; oh < B4_OH2; oh++)
  {
    for (int ow = 0; ow < B4_OW2; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * B4_STRIDE2) + kh - ALL_PADDING;
          int iw = (ow * B4_STRIDE2) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B4_OH1 && iw < B4_OW1)
          {
          conv2:
            for (int oc = 0; oc < B4_OC2; oc++)
            {
              for (int ic = 0; ic < B4_OC1; ic++)
              {
                if (bb_conv_weight[(kh * OTHER_KERNEL_SIZE * B4_OC2 * B4_OC1 / AP_SIZE) + (kw * B4_OC2 * B4_OC1 / AP_SIZE) + (oc * B4_OC1 / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_output1_acgivation[ih * B4_OW1 * B4_OC1 + iw * B4_OC1 + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_output1_acgivation[ih * B4_OW1 * B4_OC1 + iw * B4_OC1 + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
      for (int oc = 0; oc < B4_OC2; oc++)
      {
        output_activatoin[oh * B4_OW2 * B4_OC2 + ow * B4_OC2 + oc] += output_tile_oc[oc] * B4_SCALE;
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
}

void last_block(
    WEIGHT_T *bb_conv_weight,
    WEIGHT_T *bb_fc_weight,
    float *bb_conv_bn_alpha,
    float *bb_conv_bn_beta,
    float *input_activation,
    float *output_activatoin)
{

  ap_uint<1> temp_acgivation[B5_IH * B5_IW * B5_IC];
  float output_tile_oc[B5_OH * B5_OW * B5_OC] = {0};

bn_relu_last_block:
  for (int hw = 0; hw < B5_IH * B5_IW; hw++)
  {
    for (int ic = 0; ic < B5_IC; ic++)
    {
      int input_value = input_activation[hw * B5_IC + ic];
      temp_activation[hw * B5_IC + ic] = RELU(input_value * bn_weight1[ic] + bn_alpha[ic]);
      output_tile_oc[hw * B5_IC + ic] = input_value;
    }
  }

cont_last_block:
  for (int oh = 0; oh < B5_OH; oh++)
  {
    for (int ow = 0; ow < B5_OW; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = oh + kh - ALL_PADDING;
          int iw = ow + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < B5_IH && iw < B5_IW)
          {
            for (int oc = 0; oc < B5_OC; oc++)
            {
              for (int ic = 0; ic < B5_IC; ic++)
              {
                if (bb_conv_weight[(kh * OTHER_KERNEL_SIZE * B5_OC * B5_IC / AP_SIZE) 
                  + (kw * B5_OC * B5_IC / AP_SIZE) + (oc * B5_IC / AP_SIZE) 
                  + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oh * B5_OW * B5_OC + ow * B5_OC + oc] += temp_activation[ih * B5_IW * B5_IC + iw * B5_IC + ic];
                }
                else
                {
                  output_tile_oc[oh * B5_OW * B5_OC + ow * B5_OC + oc] -= temp_activation[ih * B5_IW * B5_IC + iw * B5_IC + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
      for (int oc = 0; oc < B5_OC; oc++)
      {
        output_tile_oc[oc] += output_tile_oc[oh * B5_OW * B5_OC + ow * B5_OC + oc] * bb_conv_bn_alpha[oc] + bb_conv_bn_beta[oc];
      }
    } // loop for OW
  }   // loop for OH
  for (int oc = 0; oc < B5_OC; oc++)
  {
    output_tile_oc[oc] /= (B5_OH * B5_OW);
  }

final_fc:
  for (int fc_out = 0; fc_out < IMAGES; fc_out++)
  {
    float fc_result = 0;
    for (int oc = 0; oc < B5_OC; oc++)
    {
      if (bb_fc_weight[(fc_out * B5_OC / AP_SIZE) + (oc / AP_SIZE)][(AP_SIZE - 1) - (oc % AP_SIZE)])
      {
        fc_result += output_tile_oc[oc];
      }
      else
      {
        fc_result -= output_tile_oc[oc];
      }
    }
    output_activation[fc_out] = fc_result;
  }
}

// void conv(
//   bool skip_conv,
//   WEIGHT_T *bb_conv_bn_weight,
//   WEIGHT_T *bb_conv_weight,
//   WEIGHT_T *bb_skip_conv_bn_weight,
//   float scale2,
//   float *bb_bn_alpha,
//   float *bb_bn_beta,
//   float *bb_skip_conv_bn_alpha,
//   float *bb_skip_conv_bn_beta,
//   float *bb_conv_bn_alpha,
//   float *bb_conv_bn_beta,
//   int16_t IH,
//   int16_t IW,
//   int16_t IC,
//   int16_t OH1, // 3
//   int16_t OW1, // 3
//   int16_t OC1, // 3
//   int16_t OH2, // 6
//   int16_t OW2, // 6
//   int16_t OC2, // 6
//   int16_t stride1,  // 3
//   int16_t stride2, // 6
//   float *input_activation, float *output_activatoin)
// {

//   int temp_activation[MAX_INPUT_SIZE];
//   int temp_output1_acgivation[MAX_INPUT_SIZE];
//   float output_tile_oc[TILE_SIZE] = {0};

// bn_relu0:
//   for (int hw = 0; hw < IH * IW; hw++)
//   {
//     for (int ic = 0; ic < IC; ic++)
//     {
//       temp_activation[hw * IC + ic] = RELU(input_activation[hw * IC + ic] * bb_bn_alpha[ic] + bb_bn_beta[ic]);
//     }
//   }

//   for (int oh = 0; oh < OH1; oh++)
//   {
//     for (int ow = 0; ow < OW1; ow++)
//     {
//       for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
//       {
//         for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
//         {
//           int ih = (oh * stride1) + kh - ALL_PADDING;
//           int iw = (ow * stride1) + kw - ALL_PADDING;
//           if (ih >= 0 && iw >= 0 && ih < IH && iw < IW)
//           {
//           conv1:
//             for (int oc = 0; oc < OC1; oc++)
//             {
//               for (int ic = 0; ic < IC; ic++)
//               {
//                 if (bb_conv_bn_weight[(kh * OTHER_KERNEL_SIZE * OC1 * IC / AP_SIZE) + (kw * OC1 * IC / AP_SIZE) + (oc * IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
//                 {
//                   output_tile_oc[oc] += temp_activation[ih * IW * IC + iw * IC + ic];
//                 }
//                 else
//                 {
//                   output_tile_oc[oc] -= temp_activation[ih * IW * IC + iw * IC + ic];
//                 }
//               } // loop for IC
//             }   // loop for OC
//           }
//         } // loop for KW
//       }   // loop  for KH
//     bn_relu1:
//       for (int oc = 0; oc < OC1; oc++)
//       {
//         temp_output1_acgivation[oh * OW1 * OC1 + ow * OC1 + oc] = RELU(output_tile_oc[oc] * bb_conv_bn_alpha[oc] + bb_conv_bn_beta[oc]);
//         output_tile_oc[oc] = 0; // reinitialzie to zero
//       }
//     } // loop for OW
//   }   // loop for OH
//   for (int oh = 0; oh < OH2; oh++)
//   {
//     for (int ow = 0; ow < OW2; ow++)
//     {
//       if (skip_conv)
//       {
//         for (int kh = 0; kh < SKIP_CONV_BN_KERNEL_SIZE; kh++)
//         {
//           for (int kw = 0; kw < SKIP_CONV_BN_KERNEL_SIZE; kw++)
//           {
//             int ih = (oh * SKIP_CONV_BN_STRIDE) + kh - SKIP_CONV_BN_PADDING;
//             int iw = (ow * SKIP_CONV_BN_STRIDE) + kw - SKIP_CONV_BN_PADDING;
//             if (ih >= 0 && iw >= 0 && ih < IH && iw < IW)
//             {
//             conv_skip:
//               for (int oc = 0; oc < OC2; oc++)
//               {
//                 for (int ic = 0; ic < IC; ic++)
//                 {
//                   if (bb_skip_conv_bn_weight[(kh * SKIP_CONV_BN_KERNEL_SIZE * OC2 * IC / AP_SIZE) + (kw * OC2 * IC / AP_SIZE) + (oc * IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
//                   {
//                     output_tile_oc[oc] += temp_activation[ih * IW * IC + iw * IC + ic];
//                   }
//                   else
//                   {
//                     output_tile_oc[oc] -= temp_activation[ih * IW * IC + iw * IC + ic];
//                   }
//                 } // loop for IC
//               }   // loop for OW
//             }
//           } // loop for KW
//         }   // loop  for KH
//       bn_skip:
//         for (int oc = 0; oc < OC2; oc++)
//         { // Skip conv residual
//           output_activatoin[oh * OW2 * OC2 + ow * OC2 + oc] = output_tile_oc[oc] * bb_skip_conv_bn_alpha[oc] + bb_skip_conv_bn_beta[oc];
//           output_tile_oc[oc] = 0; // reinitialzie to zero
//         }
//       }
//       else
//       { // skip add
//       skip_add:
//         for (int oc = 0; oc < OC2; oc++)
//         {
//           output_activatoin[oh * OW2 * OC2 + ow * OC2 + oc] = input_activation[oh * OW2 * OC2 + ow * OC2 + oc];
//         }
//       }
//     } // loop for OW
//   }   // loop for OH
//   for (int oh = 0; oh < OH2; oh++)
//   {
//     for (int ow = 0; ow < OW2; ow++)
//     {
//       for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
//       {
//         for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
//         {
//           int ih = (oh * stride2) + kh - ALL_PADDING;
//           int iw = (ow * stride2) + kw - ALL_PADDING;
//           if (ih >= 0 && iw >= 0 && ih < OH1 && iw < OW1)
//           {
//           conv2:
//             for (int oc = 0; oc < OC2; oc++)
//             {
//               for (int ic = 0; ic < OC1; ic++)
//               {
//                 if (bb_conv_weight[(kh * OTHER_KERNEL_SIZE * OC2 * OC1 / AP_SIZE) + (kw * OC2 * OC1 / AP_SIZE) + (oc * OC1 / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
//                 {
//                   output_tile_oc[oc] += temp_output1_acgivation[ih * OW1 * OC1 + iw * OC1 + ic];
//                 }
//                 else
//                 {
//                   output_tile_oc[oc] -= temp_output1_acgivation[ih * OW1 * OC1 + iw * OC1 + ic];
//                 }
//               } // loop for IC
//             }   // loop for OC
//           }
//         } // loop for KW
//       }   // loop  for KH
//       for (int oc = 0; oc < OC2; oc++)
//       {
//         output_activatoin[oh * OW2 * OC2 + ow * OC2 + oc] += output_tile_oc[oc] * scale2;
//         output_tile_oc[oc] = 0; // reinitialzie to zero
//       }
//     } // loop for OW
//   }   // loop for OH
// }
