#include <ap_int.h>
#include <stdint.h>

#define RELU(x) ((x) >= 0.5) ? 1 : 0
#define BIT_WIDTH 16

typedef struct {
  bool skip_conv;
  ap_uint<BIT_WIDTH> *weights1;
  ap_uint<BIT_WIDTH> *weights2;
  ap_uint<BIT_WIDTH> *weights_skip;
  float scale1;
  float scale2;
  float *bn_weight1;
  float *bn_alpha1;
  float *bn_weight_skip;
  float *bn_alpha_skip; 
  uint16_t IH;
  uint16_t IW;
  uint16_t IC;
  uint16_t OH1;
  uint16_t OW1;
  uint16_t OC1;
  uint16_t OH2;
  uint16_t OW2;
  uint16_t OC2;
  uint8_t KH1;
  uint8_t KW1;
  uint8_t KH2;
  uint8_t KW2;
  uint8_t KH_SKIP;
  uint8_t KW_SKIP;
  uint8_t padding1;
  uint8_t padding2;
  uint8_t padding_skip;
  uint8_t stride1;
  uint8_t stride2;
  uint8_t stride_skip;
} Info;

#define MAX_INUT_SIZE 80*32*32
#define TILE_SIZE 320

void basic_block(Info info, float *input_activation, float *output_activatoin) {

  ap_uint<1> temp_acgivation [MAX_INPUT_SIZE];
  ap_uint<2> temp_output1_acgivation [MAX_INPUT_SIZE];
  float output_tile_oc[TILE_SIZE] = {0};


  bn_relu0: for(int hw = 0 hw< info.IH*info.IW; hw++) {
    for(int info.IC = 0; info.IC < info.IC; info.IC++) {
      temp_activation[hw * info.IC] = RELU(input_activation[hw + info.IC] * info.bn_weight1[info.IC] + info.bn_alpha[info.IC]);
    }
  } 
   
  for(int oh = 0; oh < info.OH1; oh++) {
		for(int ow = 0; ow < info.OW1; ow++) {
			for(int kh = 0; kh < info.KH1; kh++) {
				for(int kw = 0; kw < info.KW1; kw++) {
        int ih = (oh * info.stride1) + kh - info.padding1;
        int info.IW = (ow * info.stride1) + kw - info.padding1;
        if(ih >= 0 && info.IW >= 0 && ih < info.IH && info.IW < info.IW){
          conv1: for(int oc = 0; oc < info.OC1; oc++) {
              for(int info.IC = 0; info.IC < info.IC ; info.IC++) {
                if(info.weights1[kh * KW * info.OC1 * info.IC + kw * info.OC1 * info.IC + oc * info.IC + (info.IC/32)][31 - (info.IC%32)]) {
                  // output1_activatoin[oh * info.OW1 * info.OC1 + ow * info.OC1 + oc] += temp_activation[ih * info.IW * info.IC * info.IW * info.IC + info.IC];
                  output_tile_oc[oc] += temp_activation[ih * info.IW * info.IC * info.IW * info.IC + info.IC];
                }
                else {
                  // output1_activatoin[oh * info.OW1 * info.OC1 + ow * info.OC1 + oc] -= temp_activation[ih * info.IW * info.IC * info.IW * info.IC + info.IC];
                  output_tile_oc[oc] -= temp_activation[ih * info.IW * info.IC * info.IW * info.IC + info.IC];
                }
              }// loop for info.IC
            } // loop for OW
          } 
        } // loop for KW
      } // loop  for KH
      bn_relu1: for(int oc = 0; oc < info.OC; oc ++) {
        // temp_output1_acgivation[oh * info.OW1 * info.OC1 + ow * info.OC1 + oc] 
        //   = RELU(output_activation[oh * info.OW1 * info.OC1 + ow * info.OC1 + oc] * bn_weight2[oc] * scale1 + bn_precompute_add[oc]);
        temp_output1_acgivation[oh * info.OW1 * info.OC1 + ow * info.OC1 + oc] 
          = RELU(output_tile_oc[oc] * bn_weight2[oc] * info.scale1 + bn_precompute_add[oc]);
        output_tile_oc[oc] = 0; // reinitialzie to zero
      }
    } // loop for OW
  }//loop for OH
  for(int oh = 0; oh < info.OH2; oh++) {
    for(int ow = 0; ow < info.OW2; ow++) {
      if(skip_conv) {
        for(int kh = 0; kh < info.KH_SKIP; kh++) {
          for(int kw = 0; kw < info.KW_SKIP; kw++) {
          int ih = (oh * info.stride_skip) + kh - info.padding_skip;
          int info.IW = (ow * info.stride_skip) + kw - info.padding_skip;
          if(ih >= 0 && info.IW >= 0 && ih < IH && info.IW < info.IW){
            conv_skip: for(int oc = 0; oc < info.OC2; oc++) {
                for(int info.IC = 0; info.IC < info.IC ; info.IC++) {
                  if(info.weights_skip[kh * KW * info.OC2 * info.IC + kw * info.OC2 * info.IC + oc * info.IC + (info.IC/32)][31 - (info.IC%32)]) {
                    // output_activation[oh * info.OW2 * info.OC2 + ow * info.OC2 + oc] += temp_activation[ih * info.IW * info.IC * info.IW * info.IC + info.IC];
                    output_tile_oc[oc] += temp_activation[ih * info.IW * info.IC * info.IW * info.IC + info.IC];

                  }
                  else {
                    output_tile_oc[oc] -= temp_activation[ih * info.IW * info.IC * info.IW * info.IC + info.IC];
                  }
                }// loop for info.IC
              } // loop for OW
            } 
          } // loop for KW
        } // loop  for KH
        bn_skip: for(int oc = 0; oc < info.OC2; oc ++) { // Skip conv residual 
          output_activatoin[oh * info.OW2 * info.OC2 + ow * info.OC2 + oc] 
            = output_tile_oc[oc] * bn_weight_skip[oc] * scale2 + bn_alpha_skip[oc]; 
          output_tile_oc[oc] = 0; // reinitialzie to zero
        }
      }
      else { //skip add
        skip_add:for(int oc = 0; oc < info.OC2; oc++)
          output_activatoin[oh * info.OW2 * info.OC2 + ow * info.OC2 + oc] = input_activation[oh * info.OW2 * info.OC2 * ow * info.OC2 + oc];
      } 
    }// loop for OW
  }//loop for OH

  for(int oh = 0; oh < info.OH2; oh++) {
    for(int ow = 0; ow < info.OW2; ow++) {
      for(int kh = 0; kh < info.KH2; kh++) {
        for(int kw = 0; kw < info.KW2; kw++) {
          int ih = (oh * info.stride2) + kh - info.padding2;
          int info.IW = (ow * info.stride2) + kw - info.padding2;
          if(ih >= 0 && info.IW >= 0 && ih < OH1 && info.IW < info.OW1){
            conv2: for(int oc = 0; oc < info.OC2; oc++) {
              for(int info.IC = 0; info.IC < info.OC1 ; info.IC++) {
                if(info.weights2[kh * KW * info.OC2 * info.OC1 + kw * info.OC2 * info.OC1 + oc * info.OC1 + (info.IC/32)][31 - (info.IC%32)]) {
                  output_tile_oc[oc] += temp_output1_acgivation[ih * info.OW1 * info.OC1 * info.IW * info.OC1 + info.IC];
                }
                else {
                  output_tile_oc[oc] -= temp_output1_acgivation[ih * info.OW1 * info.OC1 * info.IW * info.OC1 + info.IC];
                }
              }// loop for info.IC
            } // loop for OC
          } 
        } // loop for KW
      } // loop  for KH 
      for(int oc = 0; oc < info.OC2; oc++) {
        output_activatoin[oh * info.OW2 * info.OC2 + ow * info.OC2 + oc] += output_tile_oc[oc];
      }
    } // loop for OW
  }//loop for OH
}
