#include <fstream>
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <stdlib.h>

//#include "run.h"

int main(void) {

  int layers[5] = {0, 3, 4, 6, 7};
  std::map<int, int> to_layer_num = { {0, 0}, {3, 1}, {4, 2}, {6, 3}, {7, 4} };

  std::string root_dir = "/home/sungjun/lab/assign/any_precision/param_text/";
  std::string bn_filenames[] = {
    "layers.0.bn_alpha.out",
    "layers.0.bn_beta.out",
    "layers.0.conv_bn_alpha.out",
    "layers.0.conv_bn_beta.out",
  
    "layers.3.bn_alpha.out",
    "layers.3.bn_beta.out",
    "layers.3.skip_conv_bn_alpha.out",
    "layers.3.skip_conv_bn_beta.out",
    "layers.3.conv_bn_alpha.out",
    "layers.3.conv_bn_beta.out",

    "layers.4.bn_alpha.out",
    "layers.4.bn_beta.out",
    "layers.4.conv_bn_alpha.out",
    "layers.4.conv_bn_beta.out",

    "layers.6.bn_alpha.out",
    "layers.6.bn_beta.out",
    "layers.6.skip_conv_bn_alpha.out",
    "layers.6.skip_conv_bn_beta.out",
    "layers.6.conv_bn_alpha.out",
    "layers.6.conv_bn_beta.out",

    "layers.7.bn_alpha.out",
    "layers.7.bn_beta.out",
  };
  std::string weight_filenames[] = {
    "layers.0.conv_bn_weight.out",
    "layers.0.conv_weight.out",

    "layers.3.skip_conv_weight.out",
    "layers.3.conv_bn_weight.out",
    "layers.3.conv_weight.out",

    "layers.4.conv_bn_weight.out",
    "layers.4.conv_weight.out",

    "layers.6.skip_conv_weight.out",
    "layers.6.conv_bn_weight.out",
    "layers.6.conv_weight.out",

    "layers.7.conv_weight.out"
  };

  std::vector<std::vector<float>> bb_bn(5);
  for (int i = 0; i < sizeof(bn_filenames) / sizeof(bn_filenames[0]); ++i) {
    std::string filename = root_dir + bn_filenames[i];

    int num_layer = std::stoi(filename.substr(filename.find(".") + 1, filename.find(".") + 2));
    num_layer = to_layer_num[num_layer];

    std::vector<float> bn;
    std::ifstream infile(filename);
    float val;
    while (infile >> val) {
      bn.push_back(val);
    }
    infile.close();

    bb_bn[num_layer].insert(bb_bn[num_layer].end(), bn.begin(), bn.end());
  }
  std::vector<std::vector<uint8_t>> bb_weight(5);
  for (int i = 0; i < sizeof(weight_filenames) / sizeof(weight_filenames[0]); ++i) {
    std::string filename = root_dir + bn_filenames[i];

    int num_layer = std::stoi(filename.substr(filename.find(".") + 1, filename.find(".") + 2));
    num_layer = to_layer_num[num_layer];

    std::vector<float> weight;
    std::ifstream infile(filename);
    uint8_t val;
    while (infile >> val) {
      printf("%u\n", val);
      weight.push_back(val);
    }
    infile.close();

    bb_weight[num_layer].insert(bb_weight[num_layer].end(), weight.begin(), weight.end());
    break;
  }

  std::vector<float> input;
  std::string filename = root_dir + "input.out_";
  std::ifstream infile(filename);
  float val;
  while (infile >> val) {
    input.push_back(val);
  }
  infile.close();

  // run(&input[0],
  //     &bb_bn[0][0], &bb_weight[0][0],
  //     &bb_bn[1][0], &bb_weight[1][0],
  //     &bb_bn[2][0], &bb_weight[2][0],
  //     &bb_bn[3][0], &bb_weight[3][0],
  //     &bb_bn[4][0], &bb_weight[4][0],
  //     nullptr, nullptr, nullptr);

  return 0;
}

