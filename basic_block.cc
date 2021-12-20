#include <ap_int.h>


#define RELU(x) ((x) >= 0.5) ? 1 : 0;

void basic_block() {
  int stride, padding;
  ap_uint<32> weights1 [KH1 * KW1 * OC1* IC/32];
  ap_uint<32> weights2 [KH2 * KW2 * OC2* OC1/32];
  ap_uint<32> weights_skio [KH_SKIP * KH_SKIP * OC2* IC/32];


  float input_activation [IH * IW * IC]; 
  ap_uint<1> temp_acgivation [IH*IW*IC];
  ap_uint<2> temp_output1_acgivation [OH1*OW1*OC1];
  float output_activation [OH2*OW2*OC2];
  float bn_weight1 [IC];
  float bn_alpha1[IC];
  float bn_weight2[OC1];
  float bn_alpha2 [OC1]; 
  bn_relu0: for(int hw = 0 hw< IH*IW; hw++) {
    for(int ic = 0; ic < IC; ic++) {
      temp_activation[hw * ic] = RELU(input_activation[hw + ic] * bn_weight1[IC] + bn_alpha[IC]);
    }
  } 
   
  for(int oh = 0; oh < OH1; oh++) {
		for(int ow = 0; ow < OW1; ow++) {
			for(int kh = 0; kh < info.KH1; kh++) {
				for(int kw = 0; kw < info.KW1; kw++) {
        int ih = (oh * stride) + kh - padding;
        int iw = (ow * stride) + kw - padding;
        if(ih >= 0 && iw >= 0 && ih < info.IH && iw < info.IW){
          conv1: for(int oc = 0; oc < OC1; oc++) {
              for(int ic = 0; ic < IC1 ; ic++) {
                if(weights[kh * KW * OC1 * IC + kw * OC1 * IC + oc * IC + (ic/32)][31 - (ic%32)]) {
                  output_activation[oh * OW1 * OC1 + ow * OC1 + oc] += temp_activation[ih * IW * IC * iw * IC + ic];
                }
                else {
                  output_activation[oh * OW1 * OC1 + ow * OC1 + oc] -= temp_activation[ih * IW * IC * iw * IC + ic];
                }
              }// loop for IC
            } // loop for OW
          } 
        } // loop for KW
      } // loop  for KH
      bn_relu1: for(int oc = 0; oc < info.OC; oc ++) {
        temp_output1_acgivation[oh * OW1 * OC1 + ow * OC1 + oc] 
          = RELU(output_activation[oh * OW1 * OC1 + ow * OC1 + oc] * bn_weight2[oc] * scale1 + bn_precompute_add[oc]);
        output_activation[oh * OW1 * OC1 + ow * OC1 + oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }//loop for OH
  for(int oh = 0; oh < OH2; oh++) {
    for(int ow = 0; ow < OW2; ow++) {
      if(skip_conv) {
        for(int kh = 0; kh < KH_SKIP; kh++) {
          for(int kw = 0; kw < KW_SKIP; kw++) {
          int ih = (oh * stride) + kh - padding;
          int iw = (ow * stride) + kw - padding;
          if(ih >= 0 && iw >= 0 && ih < IH && iw < IW){
            conv_skip: for(int oc = 0; oc < OC2; oc++) {
                for(int ic = 0; ic < IC1 ; ic++) {
                  if(weights[kh * KW * OC2 * IC + kw * OC2 * IC + oc * IC + (ic/32)][31 - (ic%32)]) {
                    output_activation[oh * OW2 * OC2 + ow * OC2 + oc] += temp_activation[ih * IW * IC * iw * IC + ic];
                  }
                  else {
                    output_activation[oh * OW2 * OC2 + ow * OC2 + oc] -= temp_activation[ih * IW * IC * iw * IC + ic];
                  }
                }// loop for IC
              } // loop for OW
            } 
          } // loop for KW
        } // loop  for KH
        bn_skip: for(int oc = 0; oc < OC2; oc ++) {
          output_activation[oh * OW2 * OC2 + ow * OC2 + oc] 
            = output_activation[oh * OW2 * OC2 + ow * OC2 + oc] * bn_weight2[oc] * scale2 + bn_alpha2[oc];; 
          // Skip conv residual 
        }
      }
      else { //skip add
        for(int oc = 0; oc < OC2; oc++)
          output_activation[oh * OW2 * OC2 + ow * OC2 + oc] = temp_activation[oh * OW2 * OC2 * ow * OC2 + oc];
      } 
    }// loop for OW
  }//loop for OH

  for(int oh = 0; oh < OH2; oh++) {
    for(int ow = 0; ow < OW2; ow++) {
      for(int kh = 0; kh < KH2; kh++) {
        for(int kw = 0; kw < KW2; kw++) {
          int ih = (oh * stride) + kh - padding;
          int iw = (ow * stride) + kw - padding;
          if(ih >= 0 && iw >= 0 && ih < OH1 && iw < OW1){
            conv2: for(int oc = 0; oc < OC2; oc++) {
              for(int ic = 0; ic < OC1 ; ic++) {
                if(weights[kh * KW * OC2 * OC1 + kw * OC2 * OC1 + oc * OC1 + (ic/32)][31 - (ic%32)]) {
                  output_activation[oh * OW2 * OC2 + ow * OC2 + oc] += temp_output1_acgivation[ih * OW1 * OC1 * iw * OC1 + ic];
                }
                else {
                  output_activation[oh * OW2 * OC2 + ow * OC2 + oc] -= temp_output1_acgivation[ih * OW1 * OC1 * iw * OC1 + ic];
                }
              }// loop for IC
            } // loop for OC
          } 
        } // loop for KW
      } // loop  for KH  
    } // loop for OW
  }//loop for OH
}
