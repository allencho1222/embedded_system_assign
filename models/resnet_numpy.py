import numpy as np
import time

def reshape(x, shape, permute=False):
    out = np.unpackbits(x)
    out = out.reshape(shape)
    if permute:
        out = np.transpose(out, (0, 3, 1, 2))
    return out

def quantize_weight(data):
    E = np.mean(np.abs(data))
    weight = np.tanh(data)
    weight = weight / 2 / np.max(np.abs(weight)) +0.5
    weight_q = 2 * np.round(weight) - 1
    weight_q = weight_q * E
    return weight_q

def getWindows(input, output_size, kernel_size, padding=0, stride=1, dilate=0):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

    in_b, in_c, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )

class Conv2D:
    def __init__(self, in_channels, out_channels, weight, kernel_size=3, stride=1, padding=0, quantize=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = weight
        self.quantize = quantize

    def forward(self, x):
        if self.quantize:
            weight_q = quantize_weight(self.weight)
        else:
            weight_q = self.weight
        n, c, h, w = x.shape
        out_h = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.kernel_size + 2 * self.padding) // self.stride + 1

        windows = getWindows(x, (n, c, out_h, out_w), self.kernel_size, self.padding, self.stride)

        out = np.einsum('bihwkl,oikl->bohw', windows, weight_q)

        return out

def func_conv0(input, in_channels, out_channels, weight, kernel_size=3, stride=1, padding=0, nchw=False):
    n, c, h, w = input.shape
    out_h = (h - kernel_size + 2 * padding) // stride + 1
    out_w = (w - kernel_size + 2 * padding) // stride + 1

    windows = getWindows(input, (n, c, out_h, out_w), kernel_size, padding, stride)

    out = np.einsum('bihwkl,oikl->bohw', windows, weight)

    if nchw:
        out = out.transpose(0,2,3,1)

    return out


class Activate:
    def __init__(self):
        pass

    def forward(self, x):
        ## Relu ##
        return np.round(np.clip(x, 0, None))

class BN:
    def __init__(self, mean, var, gamma, beta):
        self.mean = mean
        self.var = var
        self.gamma = gamma
        self.beta = beta

        self.eps = 1e-05

    def forward(self, x):
        out = np.zeros(x.shape)
        for c in range(x.shape[1]):
            out[:,c,:,:] = self.gamma[c]* (x[:,c,:,:] - self.mean[c]) / (np.sqrt(self.var[c] + self.eps)) + self.beta[c]
        return out

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
    def forward(self, x):
        weight_q = quantize_weight(self.weight)
        return x@weight_q.T + self.bias


class PreActBasicBlockQ_np():
    def __init__(self, num, bit_list, in_planes, out_planes, params, stride=1, quan=False):

        ## Quantize ##
        if quan:

            self.bit_list = bit_list
            self.wbit = self.bit_list[-1]
            self.abit = self.bit_list[-1]
        
        self.params = params
        self.num = num
        self.out_planes = out_planes
        self.stride = stride
        self.in_planes = in_planes


        bn0_weight = params[f'layers.{self.num}.bn0.bn_dict.1.weight']
        bn0_bias = params[f'layers.{self.num}.bn0.bn_dict.1.bias']
        conv0_kernel = params[f'layers.{self.num}.conv0.weight']
        bn1_weight = params[f'layers.{self.num}.bn1.bn_dict.1.weight']
        bn1_bias = params[f'layers.{self.num}.bn1.bn_dict.1.bias']
        conv1_kernel = params[f'layers.{self.num}.conv1.weight']

        self.bn0 = BN(
                        bn0_weight['mean'], 
                        bn0_weight['var'], 
                        bn0_weight['param'], 
                        bn0_bias['param']
                    )
        self.act0 = Activate()
        if num != 7 :
            ### B X C X H X W ###
            self.conv0 = Conv2D(
                                    conv0_kernel['shape'][1], 
                                    conv0_kernel['shape'][0], 
                                    #reshape(conv0_kernel['param'], conv0_kernel['shape'], True), 
                                    conv0_kernel['param'], 
                                    conv0_kernel['shape'][2], 
                                    stride, 
                                    1
                                )

            self.bn1 = BN(
                            bn1_weight['mean'], 
                            bn1_weight['var'], 
                            bn1_weight['param'], 
                            bn1_bias['param']
                        )
            self.act1 = Activate()
        self.conv1 = Conv2D(
                                conv1_kernel['shape'][1], 
                                conv1_kernel['shape'][0], 
                                #reshape(conv1_kernel['param'], conv1_kernel['shape'], True), 
                                conv1_kernel['param'], 
                                conv1_kernel['shape'][2], 
                                1, 
                                1
                            )
        self.skip_conv = None
        if stride != 1:
            skip_bn_weight = params[f'layers.{self.num}.skip_bn.weight']
            skip_bn_bias = params[f'layers.{self.num}.skip_bn.bias']
            skip_conv_kernel = params[f'layers.{self.num}.skip_conv.weight']
            self.skip_conv = Conv2D(
                                    skip_conv_kernel['shape'][1],
                                    skip_conv_kernel['shape'][0],
                                    #reshape(skip_conv_kernel['param'], skip_conv_kernel['shape'] ,True),
                                    skip_conv_kernel['param'],
                                    skip_conv_kernel['shape'][2],
                                    stride,
                                    0
                                ) 
            self.skip_bn = BN(
                        skip_bn_weight['mean'],
                        skip_bn_weight['var'],
                        skip_bn_weight['param'],
                        skip_bn_bias['param']
                    )
        

    def forward(self, x):
        out = self.bn0.forward(x)
        out = self.act0.forward(out)

        if self.skip_conv is not None:
            shortcut = self.skip_conv.forward(out) # Full-precision conv
            shortcut = self.skip_bn.forward(shortcut) # Full-precision batchnorm
        else:
            shortcut = x

        if self.num != 7:
            out = self.conv0.forward(out)
            out = self.bn1.forward(out)
            out = self.act1.forward(out)
        out = self.conv1.forward(out)

        out += shortcut

        return out

class PreActResNet_np():
    def __init__(self, block, num_units, bit_list, num_classes, params, expand=5):
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]
        self.expand = expand
        self.num_classes = num_classes

        ep = self.expand

        conv0_kernel = params['conv0.weight']
        self.conv0 = Conv2D(3, 16 * ep, conv0_kernel['param'], 3, 1, 1, quantize = False)
        
        ## selected layer list ##
        layer_list = [0, 3, 4, 6, 7]
        #layer_list = [0, 3, 4]

        strides = [1] * num_units[0] + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        channels = [16 * ep] * num_units[0] + [32 * ep] * num_units[1] + [64 * ep] * num_units[2]
        in_planes = 16 * ep

        strides = list(np.array(strides)[layer_list])
        channels = list(np.array(channels)[layer_list])

        self.layers = []

        for i, (stride, channel) in enumerate(zip(strides, channels)):
            #self.layers.append(block(i, self.bit_list, in_planes, channel, params, stride))
            self.layers.append(block(layer_list[i], self.bit_list, in_planes, channel, params, stride))
            in_planes = channel

        bn_weight = params['bn.bn_dict.1.weight']
        bn_bias = params['bn.bn_dict.1.bias']
        self.bn = BN(
                        bn_weight['mean'], 
                        bn_weight['var'], 
                        bn_weight['param'], 
                        bn_bias['param']
                    )

        fc_weight = params['fc.weight']
        fc_bias = params['fc.bias']
        #self.fc = Linear(reshape(fc_weight['param'], fc_weight['shape']), fc_bias['param'] )
        self.fc = Linear(fc_weight['param'], fc_bias['param'] )
        
    def forward(self, x):
        print('start!')
        t = time.time()
        out = self.conv0.forward(x)
        print(f'conv0 complete! elapsed time : {time.time()-t}')
        np.save('out_data/conv0.npy', out)
        for layer in self.layers:
            t = time.time()
            out = layer.forward(out)
            print(f'layer{layer.num} complete! elapsed time : {time.time()-t}')
            np.save(f'out_data/layer{layer.num}.npy', out)
        out = self.bn.forward(out)
        np.save(f'out_data/bn.npy', out)
        #out = out.mean(dim=2).mean(dim=2)
        out = np.average(np.average(out, axis=2), axis=2)
        out = self.fc.forward(out)
        np.save(f'out_data/fc.npy', out)
        return out



# For CIFAR10
def resnet20q_np(bit_list, params, num_classes=10):
    return PreActResNet_np(PreActBasicBlockQ_np, [3, 3, 3], bit_list, num_classes=num_classes, params=params)









