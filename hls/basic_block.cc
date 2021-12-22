#include "conv.h"

void conv(Info info, float *input_activation, float *output_activatoin) {

  int temp_activation [MAX_INPUT_SIZE];
  int temp_output1_acgivation [MAX_INPUT_SIZE];
  float output_tile_oc[TILE_SIZE] = {0};

  bn_relu0:
  for (int hw = 0; hw < info.IH*info.IW; hw++) {
    for (int ic = 0; ic < info.IC; ic++) {
      temp_activation[hw * info.IC + ic] = RELU(input_activation[hw * info.IC + ic] * info.bb_bn_alpha[ic] + info.bb_bn_beta[ic]);
    }
  }

  for(int oh = 0; oh < info.OH1; oh++) {
	for(int ow = 0; ow < info.OW1; ow++) {
		for(int kh = 0; kh < OTHER_KERNEL_SIZE; kh++) {
			for(int kw = 0; kw < OTHER_KERNEL_SIZE; kw++) {
			  int ih = (oh * info.stride1) + kh - ALL_PADDING;
			  int iw = (ow * info.stride1) + kw - ALL_PADDING;
			  if(ih >= 0 && iw >= 0 && ih < info.IH && iw < info.IW){
				conv1:
				for(int oc = 0; oc < info.OC1; oc++) {
				  for(int ic = 0; ic < info.IC ; ic++) {
					//  count++;
					if(info.bb_conv_bn_weight[(kh * OTHER_KERNEL_SIZE * info.OC1 * info.IC / AP_SIZE) + (kw * info.OC1 * info.IC / AP_SIZE) + (oc * info.IC / AP_SIZE) + (ic/AP_SIZE)][(AP_SIZE - 1) - (ic%AP_SIZE)]) {
						output_tile_oc[oc] += temp_activation[ih * info.IW * info.IC + iw * info.IC + ic];
					} else {
					  output_tile_oc[oc] -= temp_activation[ih * info.IW * info.IC + iw * info.IC + ic];
					}
				  }// loop for info.IC
				} // loop for OC
			  } else {
				  for(int oc = 0; oc < info.OC1; oc++) {
					  for(int ic = 0; ic < info.IC; ic++){
						  if(info.bb_conv_bn_weight[(kh * OTHER_KERNEL_SIZE * info.OC1 * info.IC / AP_SIZE) + (kw * info.OC1 * info.IC / AP_SIZE) + (oc * info.IC / AP_SIZE) + (ic/AP_SIZE)][(AP_SIZE - 1) - (ic%AP_SIZE)]){
						//	  printf("padd 1\n");
						  }
						  else {
						//	  printf("padd -1\n");
						  }
					  }
				  }
			  }
			} // loop for KW
		  } // loop  for KH
		  bn_relu1:
		  for(int oc = 0; oc < info.OC1; oc ++) {
			temp_output1_acgivation[oh * info.OW1 * info.OC1 + ow * info.OC1 + oc]
			  = RELU(output_tile_oc[oc] * info.bb_conv_bn_alpha[oc] + info.bb_conv_bn_beta[oc]);
			output_tile_oc[oc] = 0; // reinitialzie to zero
          }
      } // loop for OW
  }//loop for OH
  for(int oh = 0; oh < info.OH2; oh++) {
    for(int ow = 0; ow < info.OW2; ow++) {
      if(info.skip_conv) {
        for(int kh = 0; kh < SKIP_CONV_BN_KERNEL_SIZE; kh++) {
          for(int kw = 0; kw < SKIP_CONV_BN_KERNEL_SIZE; kw++) {
          int ih = (oh * SKIP_CONV_BN_STRIDE) + kh - SKIP_CONV_BN_PADDING;
          int iw = (ow * SKIP_CONV_BN_STRIDE) + kw - SKIP_CONV_BN_PADDING;
          if(ih >= 0 && iw >= 0 && ih < info.IH && iw < info.IW){
            conv_skip: for(int oc = 0; oc < info.OC2; oc++) {
                for(int ic = 0; ic < info.IC ; ic++) {
                  if(info.bb_skip_conv_bn_weight[(kh * SKIP_CONV_BN_KERNEL_SIZE * info.OC2 * info.IC / AP_SIZE) + (kw * info.OC2 * info.IC / AP_SIZE) + (oc * info.IC / AP_SIZE) + (ic/AP_SIZE)][(AP_SIZE - 1) - (ic%AP_SIZE)]) {
                    output_tile_oc[oc] += temp_activation[ih * info.IW * info.IC + iw * info.IC + ic];

                  }
                  else {
                    output_tile_oc[oc] -= temp_activation[ih * info.IW * info.IC + iw * info.IC + ic];
                  }
                }// loop for info.IC
              } // loop for OW
            }
          } // loop for KW
        } // loop  for KH
        bn_skip: for(int oc = 0; oc < info.OC2; oc ++) { // Skip conv residual
          output_activatoin[oh * info.OW2 * info.OC2 + ow * info.OC2 + oc]
            = output_tile_oc[oc] * info.bb_skip_conv_bn_alpha[oc] + info.bb_skip_conv_bn_beta[oc];
          output_tile_oc[oc] = 0; // reinitialzie to zero
        }
      }
      else { //skip add
        skip_add:
		for(int oc = 0; oc < info.OC2; oc++) {
          output_activatoin[oh * info.OW2 * info.OC2 + ow * info.OC2 + oc] = input_activation[oh * info.OW2 * info.OC2 + ow * info.OC2 + oc];
		}
      }
    }// loop for OW
  }//loop for OH

  for(int oh = 0; oh < info.OH2; oh++) {
    for(int ow = 0; ow < info.OW2; ow++) {
      for(int kh = 0; kh < OTHER_KERNEL_SIZE; kh++) {
        for(int kw = 0; kw < OTHER_KERNEL_SIZE; kw++) {
          int ih = (oh * info.stride2) + kh - ALL_PADDING;
          int iw = (ow * info.stride2) + kw - ALL_PADDING;
          if(ih >= 0 && iw >= 0 && ih < info.OH1 && iw < info.OW1){
            conv2:
			for(int oc = 0; oc < info.OC2; oc++) {
              for(int ic = 0; ic < info.OC1 ; ic++) {
                if(info.bb_conv_weight[(kh * OTHER_KERNEL_SIZE * info.OC2 * info.OC1 / AP_SIZE) + (kw * info.OC2 * info.OC1 / AP_SIZE) + (oc * info.OC1 / AP_SIZE) + (ic/AP_SIZE)][(AP_SIZE - 1) - (ic %AP_SIZE)]) {
                	//printf("1\n");
                  output_tile_oc[oc] += temp_output1_acgivation[ih * info.OW1 * info.OC1 + iw * info.OC1 + ic];
                }
                else {
               // 	printf("-1\n");
                  output_tile_oc[oc] -= temp_output1_acgivation[ih * info.OW1 * info.OC1 + iw * info.OC1 + ic];
                }
              }// loop for info.IC
            } // loop for OC
          }
        } // loop for KW
      } // loop  for KH
      for(int oc = 0; oc < info.OC2; oc++) {
    	//printf("%f\n", output_tile_oc[oc]);
        output_activatoin[oh * info.OW2 * info.OC2 + ow * info.OC2 + oc] += output_tile_oc[oc] * info.conv_scale;
       // printf("%f\n", output_tile_oc[oc] * info.conv_scale);
        output_tile_oc[oc] = 0; // reinitialzie to zero

      }
    } // loop for OW
  }//loop for OH
}
void last_block(Info info, float *input_activation, float *output_activatoin) {

 ap_uint<1> temp_acgivation [320*8*8];
 ap_uint<1> temp_output1_acgivation [320*8*8];
 float output_tile_oc[8*8*320] = {0};

 bn_relu_last_block: 
 for(int hw = 0 hw< 64; hw++) {
   for(int ic = 0; ic < 320; ic++) {
     int input_value = input_activation[hw + 320 + ic];
     temp_activation[hw * 320 + ic] = RELU( input_value * info.bn_weight1[ic] + info.bn_alpha[ic]);
    output_tile_oc[hw * 320 * ic] = input_value;
   }
 }

 cont_last_block: for(int oh = 0; oh < 8; oh++) {
   for(int ow = 0; ow < 8; ow++) {
     for(int kh = 0; kh < OTHER_KERNEL_SIZE; kh++) {
       for(int kw = 0; kw < OTHER_KERNEL_SIZE; kw++) {
         int ih = oh  + kh - ALL_PADDING;
         int iw = ow + kw - ALL_PADDING;
         if(ih >= 0 && iw >= 0 && ih < 8 && iw < 8){
           for(int oc = 0; oc < 320; oc++) {
             for(int ic = 0; ic < 320 ; ic++) {
               if(info.bb_conv_weight[(kh * OTHER_KERNEL_SIZE * 320 * 320 / AP_SIZE) + (kw * 320 * 320/ AP_SIZE) + (oc * 320 / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)]) {
                 output_tile_oc[oh * 8 * 320 + ow * 320 + oc] += temp_activation[ih * 8 * 320 + iw * 320 + ic];
               }
               else {
                 output_tile_oc[oh * 8 * 320 + ow * 320 + oc] -= temp_activation[ih * info.IW * info.IC + iw * info.IC + ic];
               }
             }// loop for info.IC
           } // loop for OC
         }
       } // loop for KW
     } // loop  for KH
     for(int oc = 0; oc < 320; oc++) {
       output_tile_oc[oc] += output_tile_oc[oh * 8 * 320 + ow * 320 + oc] * info.bb_skip_conv_bn_alpha[oc] + info.bb_skip_conv_bn_beta[oc];
     }
   } // loop for OW
 }//loop for OH
 for(int oc = 0; oc < 320; oc++) {
   output_tile_oc[oc] /= 64;
 }
 final_fc:
 for(int fc_out = 0; fc_out < 10; fc_out++) {
   float fc_result = 0;
   for(int oc = 0; oc < 320; oc++) {
      if(info.bb_conv_weight[(fc_out * 320 / AP_SIZE) + (oc / AP_SIZE)][(AP_SIZE - 1) - (oc % AP_SIZE)]) {
        fc_result += output_tile_oc[oc];
      }
      else {
        fc_result -= output_tile_oc[oc];
      }
   }
   output_activation[fc_out] = fc_result;
 }
}


