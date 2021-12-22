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
 ap_uint<2> temp_output1_acgivation [320*8*8];
 float output_tile_oc[320] = {0};

 bn_relu_last_block: 
 for(int hw = 0 hw< info.IH*info.IW; hw++) {
   for(int ic = 0; ic < info.IC; ic++) {
     temp_activation[hw * info.IC + ic] = RELU(input_activation[hw + info.IC + ic] * info.bn_weight1[ic] + info.bn_alpha[ic]);
   }
 }

 cont_last_block: for(int oh = 0; oh < info.OH2; oh++) {
   for(int ow = 0; ow < info.OW2; ow++) {
     for(int kh = 0; kh < OTHER_KERNEL_SIZE; kh++) {
       for(int kw = 0; kw < OTHER_KERNEL_SIZE; kw++) {
         int ih = (oh * info.stride2) + kh - ALL_PADDING;
         int info.IW = (ow * info.stride2) + kw - ALL_PADDING;
         if(ih >= 0 && info.IW >= 0 && ih < info.IH && iw < info.IW){
           for(int oc = 0; oc < info.OC2; oc++) {
             for(int ic = 0; ic < info.IC ; ic++) {
               if(info.bb_conv_weight[(kh * OTHER_KERNEL_SIZE * info.OC2 * info.IC / AP_SIZE) + (kw * info.OC2 * info.IC / AP_SIZE) + (oc * info.IC / AP_SIZE) + (ic / AP_SIZE)][(AP_SIZE - 1) - (ic % AP_SIZE)]) {
                 output_tile_oc[oc] += temp_activation[ih * info.IW * info.IC + iw * info.IC + ic];
               }
               else {
                 output_tile_oc[oc] -= temp_activation[ih * info.IW * info.IC + iw * info.IC + ic];
               }
             }// loop for info.IC
           } // loop for OC
         }
       } // loop for KW
     } // loop  for KH
     for(int oc = 0; oc < info.OC2; oc++) {
       output_activatoin[oh * info.OW2 * info.OC2 + ow * info.OC2 + oc] += output_tile_oc[oc] * info.conv_scale;
     }
   } // loop for OW
 }//loop for OH
}


