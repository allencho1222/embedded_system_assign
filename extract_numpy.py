import pickle
import argparse
import torch
import numpy as np
import math

import models

def quantize_weight(data):
    E = torch.mean(torch.abs(data)).detach()
    weight = torch.tanh(data)
    weight = weight / 2 / torch.max(torch.abs(weight)) +0.5
    weight_q = 2 * torch.round(weight) - 1
    weight_q = weight_q * E
    return weight_q

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', default = 'pretrain/114_epoch_1_bit.pth.tar')
    parser.add_argument('--save', action = 'store_true', default = False)
    args = parser.parse_args()

    # Load pretrained parameters
    checkpoint = torch.load(args.pretrain, map_location = torch.device('cpu'))

    # Init weights
    model = models.__dict__['resnet20q']([1], 10)
    model.load_state_dict(checkpoint['state_dict'])

    # Collect BN statistics
    bn = {}
    for i, x in enumerate(model.modules()):
        if isinstance(x, torch.nn.BatchNorm2d):
            bn[i] = {}
            bn[i]['param'] = x.weight.detach().numpy()
            bn[i]['mean'] = x.running_mean.numpy()
            bn[i]['var'] = x.running_var.numpy()
            bn[i]['eps'] = x.eps

    params = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        params[name] = {}
        # Module list (module list always contains quantized weights for conv)
        if 'layers' in name:
            layer_num = int(name.split('.')[1])
            # Ignore 1, 2, 5, 8 layers
            if not layer_num in [1, 2, 5, 8]:
                if 'conv' in name:
                    # We skip this layer
                    if 'conv0' in name and layer_num == 7:
                        continue
                    '''
                    quant_weight = quantize_weight(param.data)
                    # Get scale factor
                    scale = torch.abs(quant_weight[0][0][0][0]).item()
                    # Get binary value
                    quant_weight = quant_weight.apply_(lambda x : False if x <= 0.0 else True)
                    # From NCHW to NHWC
                    quant_weight = quant_weight.permute(0, 2, 3, 1)
                    # To numpy array
                    quant_weight = quant_weight.numpy().astype(np.uint8)
                    params[name]['shape'] = quant_weight.shape
                    # Pack bits
                    quant_weight = np.packbits(quant_weight.flatten())
                    params[name]['param'] = quant_weight
                    params[name]['scale'] = scale
                    '''
                    params[name]['shape'] = param.data.shape
                    params[name]['param'] = param.data.numpy()
                # batchnorm
                else:
                    # We skip this layer
                    if 'bn1' in name and layer_num == 7:
                        continue
                    if 'bn' in name and 'weight' in name:
                        params[name]['param'] = param.data.numpy()
                        params[name]['shape'] = params[name]['param'].shape
                        # Find matching bn statistics
                        did = True
                        for idx, bn_param in bn.items():
                            if np.array_equal(bn_param['param'], params[name]['param']):
                                params[name]['mean'] = bn_param['mean']
                                params[name]['var'] = bn_param['var']
                                params[name]['eps'] = bn_param['eps']
                                did = True
                                break
                        assert did
                    else: # bn bias
                        params[name]['param'] = param.data.numpy()
                        params[name]['shape'] = params[name]['param'].shape
            else:
                del params[name]
        else: # bn and fc
            if 'bn' in name and 'weight' in name: # bn weight
                params[name]['param'] = param.data.numpy()
                params[name]['shape'] = params[name]['param'].shape
                # Find matching bn statistics
                did = True
                for idx, bn_param in bn.items():
                    if np.array_equal(bn_param['param'], params[name]['param']):
                        params[name]['mean'] = bn_param['mean']
                        params[name]['var'] = bn_param['var']
                        params[name]['eps'] = bn_param['eps']
                        did = True
                        break
                assert did
            else: 
                if 'fc' in name and 'weight' in name: # fc
                    '''
                    quant_weight = quantize_weight(param.data)
                    # Get scale factor
                    scale = torch.abs(quant_weight[0][0]).item()
                    # Get binary value
                    quant_weight = quant_weight.apply_(lambda x : False if x <= 0.0 else True)
                    # To numpy array
                    quant_weight = quant_weight.numpy().astype(np.uint8)
                    params[name]['shape'] = quant_weight.shape
                    # Pack bits
                    quant_weight = np.packbits(quant_weight.flatten())
                    params[name]['param'] = quant_weight
                    params[name]['scale'] = scale
                    '''
                    params[name]['shape'] = param.data.shape
                    params[name]['param'] = param.data.numpy()
                else: # bn bias
                    params[name]['param'] = param.data.numpy()
                    params[name]['shape'] = params[name]['param'].shape

    #print(params)

    if args.save:
        with open('params_np.pkl', 'wb') as p:
            pickle.dump(params, p, protocol = pickle.HIGHEST_PROTOCOL)
