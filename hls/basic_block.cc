#include "conv.h"

void conv(
  bool skip_conv,
  ap_uint<AP_SIZE> *bb_conv_bn_weight,
  ap_uint<AP_SIZE> *bb_conv_weight,
  ap_uint<AP_SIZE> *bb_skip_conv_bn_weight,
  float scale2,
  float *bb_bn_alpha,
  float *bb_bn_beta,
  float *bb_skip_conv_bn_alpha,
  float *bb_skip_conv_bn_beta;,
  uint16_t IH,
  uint16_t IW,
  uint16_t IC,
  uint16_t OH1, // 3
  uint16_t OW1, // 3
  uint16_t OC1, // 3
  uint16_t OH2, // 6
  uint16_t OW2, // 6
  uint16_t OC2, // 6
  uint8_t stride1,  // 3
  uint8_t stride2, // 6
  float *input_activation, float *output_activatoin)
{

  int temp_activation[MAX_INPUT_SIZE];
  int temp_output1_acgivation[MAX_INPUT_SIZE];
  float output_tile_oc[TILE_SIZE] = {0};

bn_relu0:
  for (int hw = 0; hw < IH * IW; hw++)
  {
    for (int ic = 0; ic < IC; ic++)
    {
      temp_activation[hw * IC + ic] = RELU(input_activation[hw * IC + ic] * bb_bn_alpha[ic] + bb_bn_beta[ic]);
    }
  }

  for (int oh = 0; oh < OH1; oh++)
  {
    for (int ow = 0; ow < OW1; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * stride1) + kh - ALL_PADDING;
          int iw = (ow * stride1) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < IH && iw < IW)
          {
          conv1:
            for (int oc = 0; oc < OC1; oc++)
            {
              for (int ic = 0; ic < IC; ic++)
              {
                if (bb_conv_bn_weight[(kh * OTHER_KERNEL_SIZE * OC1 * IC / AP_SIZE) + (kw * OC1 * IC / AP_SIZE) + (oc * IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_activation[ih * IW * IC + iw * IC + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_activation[ih * IW * IC + iw * IC + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
    bn_relu1:
      for (int oc = 0; oc < OC1; oc++)
      {
        temp_output1_acgivation[oh * OW1 * OC1 + ow * OC1 + oc] = RELU(output_tile_oc[oc] * bb_conv_bn_alpha[oc] + bb_conv_bn_beta[oc]);
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
  for (int oh = 0; oh < OH2; oh++)
  {
    for (int ow = 0; ow < OW2; ow++)
    {
      if (skip_conv)
      {
        for (int kh = 0; kh < SKIP_CONV_BN_KERNEL_SIZE; kh++)
        {
          for (int kw = 0; kw < SKIP_CONV_BN_KERNEL_SIZE; kw++)
          {
            int ih = (oh * SKIP_CONV_BN_STRIDE) + kh - SKIP_CONV_BN_PADDING;
            int iw = (ow * SKIP_CONV_BN_STRIDE) + kw - SKIP_CONV_BN_PADDING;
            if (ih >= 0 && iw >= 0 && ih < IH && iw < IW)
            {
            conv_skip:
              for (int oc = 0; oc < OC2; oc++)
              {
                for (int ic = 0; ic < IC; ic++)
                {
                  if (bb_skip_conv_bn_weight[(kh * SKIP_CONV_BN_KERNEL_SIZE * OC2 * IC / AP_SIZE) + (kw * OC2 * IC / AP_SIZE) + (oc * IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                  {
                    output_tile_oc[oc] += temp_activation[ih * IW * IC + iw * IC + ic];
                  }
                  else
                  {
                    output_tile_oc[oc] -= temp_activation[ih * IW * IC + iw * IC + ic];
                  }
                } // loop for IC
              }   // loop for OW
            }
          } // loop for KW
        }   // loop  for KH
      bn_skip:
        for (int oc = 0; oc < OC2; oc++)
        { // Skip conv residual
          output_activatoin[oh * OW2 * OC2 + ow * OC2 + oc] = output_tile_oc[oc] * bb_skip_conv_bn_alpha[oc] + bb_skip_conv_bn_beta[oc];
          output_tile_oc[oc] = 0; // reinitialzie to zero
        }
      }
      else
      { // skip add
      skip_add:
        for (int oc = 0; oc < OC2; oc++)
        {
          output_activatoin[oh * OW2 * OC2 + ow * OC2 + oc] = input_activation[oh * OW2 * OC2 + ow * OC2 + oc];
        }
      }
    } // loop for OW
  }   // loop for OH

  for (int oh = 0; oh < OH2; oh++)
  {
    for (int ow = 0; ow < OW2; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = (oh * stride2) + kh - ALL_PADDING;
          int iw = (ow * stride2) + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < OH1 && iw < OW1)
          {
          conv2:
            for (int oc = 0; oc < OC2; oc++)
            {
              for (int ic = 0; ic < OC1; ic++)
              {
                if (bb_conv_weight[(kh * OTHER_KERNEL_SIZE * OC2 * OC1 / AP_SIZE) + (kw * OC2 * OC1 / AP_SIZE) + (oc * OC1 / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oc] += temp_output1_acgivation[ih * OW1 * OC1 + iw * OC1 + ic];
                }
                else
                {
                  output_tile_oc[oc] -= temp_output1_acgivation[ih * OW1 * OC1 + iw * OC1 + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
      for (int oc = 0; oc < OC2; oc++)
      {
        output_activatoin[oh * OW2 * OC2 + ow * OC2 + oc] += output_tile_oc[oc] * conv_scale;
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }   // loop for OH
}

void last_block(ap_int<AP_SIZE> *bb_conv_weight, float *bb_conv_bn_alpha, float *bb_conv_bn_beta, float *input_activation, float *output_activatoin)
{

  ap_uint<1> temp_acgivation[320 * 8 * 8];
  ap_uint<1> temp_output1_acgivation[320 * 8 * 8];
  float output_tile_oc[8 * 8 * 320] = {0};

bn_relu_last_block:
  for (int hw = 0 hw < 64; hw++)
  {
    for (int ic = 0; ic < 320; ic++)
    {
      int input_value = input_activation[hw + 320 + ic];
      temp_activation[hw * 320 + ic] = RELU(input_value * bn_weight1[ic] + bn_alpha[ic]);
      output_tile_oc[hw * 320 * ic] = input_value;
    }
  }

cont_last_block:
  for (int oh = 0; oh < 8; oh++)
  {
    for (int ow = 0; ow < 8; ow++)
    {
      for (int kh = 0; kh < OTHER_KERNEL_SIZE; kh++)
      {
        for (int kw = 0; kw < OTHER_KERNEL_SIZE; kw++)
        {
          int ih = oh + kh - ALL_PADDING;
          int iw = ow + kw - ALL_PADDING;
          if (ih >= 0 && iw >= 0 && ih < 8 && iw < 8)
          {
            for (int oc = 0; oc < 320; oc++)
            {
              for (int ic = 0; ic < 320; ic++)
              {
                if (bb_conv_weight[(kh * OTHER_KERNEL_SIZE * 320 * 320 / AP_SIZE) + (kw * 320 * 320 / AP_SIZE) + (oc * 320 / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)])
                {
                  output_tile_oc[oh * 8 * 320 + ow * 320 + oc] += temp_activation[ih * 8 * 320 + iw * 320 + ic];
                }
                else
                {
                  output_tile_oc[oh * 8 * 320 + ow * 320 + oc] -= temp_activation[ih * 8 * 320+ iw * 320 + ic];
                }
              } // loop for IC
            }   // loop for OC
          }
        } // loop for KW
      }   // loop  for KH
      for (int oc = 0; oc < 320; oc++)
      {
        output_tile_oc[oc] += output_tile_oc[oh * 8 * 320 + ow * 320 + oc] * bb_conv_bn_alpha[oc] + bb_conv_bn_beta[oc];
      }
    } // loop for OW
  }   // loop for OH
  for (int oc = 0; oc < 320; oc++)
  {
    output_tile_oc[oc] /= 64;
  }

final_fc:
  for (int fc_out = 0; fc_out < 10; fc_out++)
  {
    float fc_result = 0;
    for (int oc = 0; oc < 320; oc++)
    {
      if (bb_conv_weight[(fc_out * 320 / AP_SIZE) + (oc / AP_SIZE)][(AP_SIZE - 1) - (oc % AP_SIZE)])
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
