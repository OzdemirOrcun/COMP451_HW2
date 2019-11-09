from builtins import object
import numpy as np

from comp451.layers import *
from comp451.fast_layers import *
from comp451.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - leakyrelu - 2x2 max pool - affine - leakyrelu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 alpha=1e-3, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - alpha: negative slope of Leaky ReLU layers
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.alpha = alpha

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        C,H,W = input_dim
        
        F = num_filters
        filter_h = filter_size
        filter_w = filter_size
        stride_convol = 1
        P = (filter_size -1) / 2
        Hc = int((H + 2 * P - filter_h) / stride_convol) + 1
        Wc = int((W + 2 * P - filter_w) / stride_convol) + 1
        
        W1 = weight_scale * np.random.randn(F,C,filter_h,filter_w)
        b1 = np.zeros(F)
        
        pool_w,pool_h,pool_s = 2,2,2
        Hp = int((Hc - pool_h) / pool_s) + 1
        Wp = int((Wc - pool_w) / pool_s) + 1
        
        
        Hh = hidden_dim
        W2 = weight_scale * np.random.randn(F * Hp * Wp, Hh)
        b2 = np.zeros(Hh)
        
        
        
        Hc = num_classes

        W3 = weight_scale * np.random.randn(Hh, Hc)

        b3 = np.zeros(Hc)


        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['W3'] = W3
        self.params['b1'] = b1 
        self.params['b2'] = b2 
        self.params['b3'] = b3
       

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in comp451/fast_layers.py and  #
        # comp451/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        lrelu_param = {}
        lrelu_param['alpha'] = self.alpha

        conv_layer, cache_conv_layer = conv_lrelu_pool_forward(
            X,W1,b1,conv_param = conv_param,pool_param = pool_param,lrelu_param = lrelu_param)
        
        N, F, Hp, Wp = conv_layer.shape
        x = conv_layer.reshape((N, F * Hp * Wp))
        
        hidden_layer,cache_hidden_layer = affine_lrelu_forward(x,W2,b2,lrelu_param = lrelu_param)
        N, Hh = hidden_layer.shape
        
        scores,cache_scores = affine_forward(hidden_layer,W3,b3)
        
       

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        L2_reg = np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)
        data_loss, dscores = softmax_loss(scores, y)
        loss_ = 0.5 * self.reg * L2_reg
        loss = data_loss + loss_
        

        grads = {}
        dx3, dW3, db3 = affine_backward(dscores, cache_scores)
        dW3 =  dW3 + self.reg * W3
        
        dx2, dW2, db2 = affine_lrelu_backward(dx3, cache_hidden_layer)
        dW2 = dW2 + self.reg * W2
        
        dx2 = dx2.reshape(N, F, Hp, Wp)
        dx, dW1, db1 = conv_lrelu_pool_backward(dx2, cache_conv_layer)
        dW1 = dW1 + self.reg * W1

        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['W3'] = dW3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
