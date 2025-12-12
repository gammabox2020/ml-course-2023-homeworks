import numpy as np
#print(__file__)
class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`:

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

        gradInput = module.backward(input, gradOutput)
    """
    def __init__ (self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, input):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(input)

    def backward(self,input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput


    def updateOutput(self, input):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        Make sure to both store the data in `output` field and return it.
        """

        # The easiest case:

        # self.output = input
        # return self.output

        pass

    def updateGradInput(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.

        Make sure to both store the gradients in `gradInput` field and return it.
        """

        # The easiest case:

        # self.gradInput = gradOutput
        # return self.gradInput

        pass

    def accGradParameters(self, input, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"
class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially.

         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`.
    """

    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})


        Just write a little loop.
        """

        # {{ Your code goes here. ################################################
        output = input
        for module in self.modules:
            output = module.forward(output)
        self.output = output
        # }} Your code goes here. ################################################

        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)


        !!!

        To ech module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.

        !!!

        """
        # {{ Your code goes here. ################################################

        g_ = self.modules[1].backward(self.modules[0].output , gradOutput)  # gradOutput от функции потерь

        gradInput = self.modules[0].backward(input, g_)  # g_ есть  gradOutput для первого слоя

        self.gradInput = gradInput

        # {{Your code goes here. ################################################

        return self.gradInput


    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self,x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()
class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization
        stdv = 1./np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        # Your code goes here. ################################################
        # self.output = ...
        # self.output = np.dot(input, self.W.T) + self.b
        self.output = input @ self.W.T + self.b
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        # self.gradInput = ...
        self.gradInput = gradOutput @ self.W
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        # Your code goes here. ################################################
        # self.gradW = ... ; self.gradb = ...
        self.gradW += gradOutput.T @ input
        # self.gradb += gradOutput.sum(axis = 0)
        self.gradb = np.sum(gradOutput, axis=0)
        pass

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q
class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))

        # Your code goes here. ################################################
        self.output = np.exp(self.output)/np.sum(np.exp(self.output), axis=1, keepdims=True)

        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################

        stmp = self.output
        self.gradInput = stmp * (gradOutput - np.sum(gradOutput * stmp, axis=1, keepdims=True))

        return self.gradInput

    def __repr__(self):
        return "SoftMax"
class LogSoftMax(Module):
    def __init__(self):
         super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))

        # Your code goes here. ################################################

        sum_e = np.sum(np.exp(self.output), axis=1, keepdims=True)
        log_sum_e = np.log(sum_e)

        self.output = self.output - log_sum_e

        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################

        s_1 = np.exp(self.output)
        sum_g = np.sum(gradOutput, axis=1, keepdims=1)
        self.gradInput = gradOutput - s_1 * sum_g

        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"
class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        # Your code goes here. ################################################
        # use self.EPS please

        if self.moving_mean is None or self.moving_variance is None:
            self.moving_mean = 0
            self.moving_variance = 1

        if self.training == False:
            self.output = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        else:
            self.mean = np.mean(input, axis=0, keepdims=True)
            self.variance = np.var(input, axis=0, keepdims=True)
            self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * self.mean
            self.moving_variance = self.alpha * self.moving_variance + (1 - self.alpha) * np.var(input, ddof=1, axis=0)
            # c дисперсией батча
            self.output = (input - self.mean) / np.sqrt(self.variance + self.EPS)

        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################

        variance = self.variance

        Ndim = input.shape[0]

        g_input_c1 = Ndim * gradOutput
        g_input_c2 = np.sum(gradOutput, axis=0, keepdims=True)
        g_input_c3 = self.output * np.sum(gradOutput * self.output, axis=0, keepdims=True)

        self.gradInput = g_input_c1 - g_input_c2 - g_input_c3
        self.gradInput *= (1 / (Ndim * np.sqrt(variance + self.EPS)))

        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"
class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = \gamma * x + \beta
       where \gamma, \beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput*input, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"
class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, input):
        # Your code goes here. ################################################

        if self.training:
            self.mask = np.random.rand(*input.shape) > self.p

            self.output = input * self.mask / (1 - self.p)
        else:
            self.output = input

        return  self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################

        if self.training:
            self.gradInput = gradOutput * self.mask / (1.0 - self.p)
        else:
            self.gradInput = gradOutput

        return self.gradInput

    def __repr__(self):
        return "Dropout"
class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"
class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()

        self.slope = slope

    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = np.where(input > 0, input, input * self.slope)
        return  self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.where(input > 0, gradOutput, gradOutput * self.slope)
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"
class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def updateOutput(self, input):
        # Your code goes here. ################################################
        self.output = np.where(input > 0, input, self.alpha * (np.exp(input) - 1.0))
        return  self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        self.gradInput = np.where(input > 0, gradOutput, self.alpha * np.exp(input) * gradOutput)
        return self.gradInput

    def __repr__(self):
        return "ELU"
class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, input):
        # Your code goes here. ################################################
        # self.output = np.where(input > 0, input, self.alpha * (np.exp(input) - 1))
        self.output = np.log(1 + np.exp(input))
        return  self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        # self.gradInput = np.where(input > 0, 1, self.alpha * np.exp(input)) * gradOutput
        self.gradInput = gradOutput/(1 + np.exp(-input))
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"
class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None

    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)

    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)

    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output

    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Criterion"
class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def updateOutput(self, input, target):
        self.output = np.sum(np.power(input - target,2)) / input.shape[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput

    def __repr__(self):
        return "MSECriterion"
class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15
    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()

    def updateOutput(self, input, target):

        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Your code goes here. ################################################

        self.output = - np.mean( np.sum(target * np.log(input_clamp), axis=1) )

        return self.output

    def updateGradInput(self, input, target):

        # Use this trick to avoid numerical errors
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)

        # Your code goes here. ################################################

        N = input.shape[0]
        self.gradInput = (- 1 / N) *(target / input_clamp)

        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterionUnstable"
class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()

    def updateOutput(self, input, target):
        # Your code goes here. ################################################
        N, _ = input.shape
        self.output = -np.sum(input * target) / N
        # self.output = -np.mean(np.sum(input * target))
        return self.output

    def updateGradInput(self, input, target):
        # Your code goes here. ################################################
        N, _ = input.shape
        self.gradInput = -target / N
        return self.gradInput

    def __repr__(self):
        return "ClassNLLCriterion"
def sgd_momentum(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('accumulated_grads', {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):

            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))

            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)

            current_var -= old_grad
            var_index += 1
def adam_optimizer(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)   # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()

    var_index = 0
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))

            # <YOUR CODE> #######################################
            # update `current_var_first_moment`, `var_second_moment` and `current_var` values
            #np.add(... , out=var_first_moment)
            #np.add(... , out=var_second_moment)
            #current_var -= ...

            # 1 moment
            np.add(config['beta1'] * var_first_moment, (1 - config['beta1']) * current_grad, out=var_first_moment)

            # 2 moment
            np.add(config['beta2'] * var_second_moment, (1 - config['beta2']) * (current_grad ** 2), out=var_second_moment)

            current_var -= lr_t * var_first_moment / (np.sqrt(var_second_moment) + config['epsilon'])

            # small checks that you've updated the state; use np.add for rewriting np.arrays values
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1

import scipy as sp
import scipy.signal
#import skimage

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size

        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        pad_size = self.kernel_size // 2
        # YOUR CODE ##############################
        # 1. zero-pad the input array
        # 2. compute convolution using scipy.signal.correlate(... , mode='valid')
        # 3. add bias value

        # self.output = ...

        padded_input = np.pad(input,
                              ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                              mode='constant')

        dim_N, _, dim_H, dim_W = input.shape

        self.output = np.zeros((dim_N, self.out_channels, dim_H, dim_W), dtype=input.dtype)

        for n in range(dim_N):
            x_n = padded_input[n]
            for c_out in range(self.out_channels):
                corr = sp.signal.correlate(
                    x_n, self.W[c_out],
                    mode='valid'
                )
                self.output[n, c_out] = corr[0] + self.b[c_out]

        return self.output

    def updateGradInput(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        # YOUR CODE ##############################
        # 1. zero-pad the gradOutput
        # 2. compute 'self.gradInput' value using scipy.signal.correlate(... , mode='valid')

        # self.gradInput = ...

        padded = np.pad(gradOutput,
                                   ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                                   mode='constant')

        gradInput = np.zeros_like(input)
        for batch in range(input.shape[0]):
            for in_ch in range(self.in_channels):
                for out_ch in range(self.out_channels):
                    gradInput[batch, in_ch] += sp.signal.correlate(padded[batch, out_ch],
                                                                   self.W[out_ch, in_ch][::-1, ::-1],
                                                                   mode='valid')

        self.gradInput = gradInput
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        pad_size = self.kernel_size // 2
        # YOUR CODE #############
        # 1. zero-pad the input
        # 2. compute 'self.gradW' using scipy.signal.correlate(... , mode='valid')
        # 3. compute 'self.gradb' - formulas like in Linear of ChannelwiseScaling layers

        # self.gradW = ...
        # self.gradb = ...
        padded = np.pad(input,
                              ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                              mode='constant')

        for out_ch in range(self.out_channels):
            for in_ch in range(self.in_channels):
                gradW_sum = np.zeros_like(self.gradW[out_ch, in_ch])
                for batch in range(input.shape[0]):
                    gradW_sum += sp.signal.correlate(padded[batch, in_ch],
                                                     gradOutput[batch, out_ch],
                                                     mode='valid')
                self.gradW[out_ch, in_ch] = gradW_sum

        self.gradb = gradOutput.sum(axis=(0, 2, 3))

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Conv2d %d -> %d' %(s[1],s[0])
        return q
class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None

    def updateOutput(self, input):
        input_h, input_w = input.shape[-2:]
        # your may remove these asserts and implement MaxPool2d with padding
        assert input_h % self.kernel_size == 0
        assert input_w % self.kernel_size == 0

        # YOUR CODE #############################
        # self.output = ...
        # self.max_indices = ...

        batch_size, num_channels, input_h, input_w = input.shape

        out_h = input_h // self.kernel_size
        out_w = input_w // self.kernel_size

        self.output = np.zeros((batch_size, num_channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, num_channels, out_h, out_w, 2), dtype=int)

        # прямой проход
        for i in range(out_h):
            for j in range(out_w):
                region = input[:, :, i*self.kernel_size:(i+1)*self.kernel_size, j*self.kernel_size:(j+1)*self.kernel_size]

                max_vals = np.max(region, axis=(2, 3))
                self.output[:, :, i, j] = max_vals

                max_indices_flat = np.argmax(region.reshape(batch_size, num_channels, -1), axis=-1)
                max_i, max_j = np.unravel_index(max_indices_flat, (self.kernel_size, self.kernel_size))

                self.max_indices[:, :, i, j, 0] = max_i

                self.max_indices[:, :, i, j, 1] = max_j
                #self.max_indices[:, :, i, j] = max_indices_flat

        return self.output

    def updateGradInput(self, input, gradOutput):
        # YOUR CODE #############################
        # self.gradInput = ...

        dimN, dimC, dimH, dimW = input.shape

        self.gradInput = np.zeros_like(input)

        for n in range(dimN):
            for c in range(dimC):
                for i in range(dimH // self.kernel_size):
                    for j in range(dimW // self.kernel_size):
                        idx = self.max_indices[n, c, i, j]

                        # h_idx = i * self.kernel_size + idx // self.kernel_size
                        # w_idx = j * self.kernel_size + idx % self.kernel_size

                        # self.gradInput[n, c, h_idx, w_idx] += gradOutput[n, c, i, j]

                        di = self.max_indices[n, c, i, j, 0]
                        dj = self.max_indices[n, c, i, j, 1]
                        h_idx = i * self.kernel_size + di
                        w_idx = j * self.kernel_size + dj
                        self.gradInput[n, c, h_idx, w_idx] += gradOutput[n, c, i, j]

        return self.gradInput

    def __repr__(self):
        q = 'MaxPool2d, kern %d, stride %d' %(self.kernel_size, self.kernel_size)
        return q
class Flatten(Module):
    def __init__(self):
         super(Flatten, self).__init__()

    def updateOutput(self, input):
        self.output = input.reshape(len(input), -1)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput.reshape(input.shape)
        return self.gradInput

    def __repr__(self):
        return "Flatten"

