from keras import backend as K
from keras.layers import Convolution2D
from keras import activations


class MaxMinConvolution2D(Convolution2D):
    """MaxMinConvolution Layer for spatial data such as images. It inherits
       the Convolution2D Layer. Each Keras Layer has to implement three 
       methods which are called at appropriate times while fit / predict etc.

        Parameters
        ==========

        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.

        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
    """

    def __init__(self, nb_filter, nb_row, nb_col, activation=None, **kwargs):
        """Initialize this layer."""

        # To be applied after concatenation, while activation of internal 
        # Convolution layer is linear activation.
        self.activation_ = activations.get(activation)
        
        super(MaxMinConvolution2D, self).__init__(nb_filter, nb_row, nb_col, **kwargs)

    def build(self, input_shape):
        super(MaxMinConvolution2D, self).build(input_shape)

    def call(self, x, mask=None):
        """The operations performed by this layer lie in this method. This layer
           simply takes the output of Convolution2D (which is a 4D tensor), and 
           concatenates it with its own negative copy. Concatenation occurs along
           the axis of channels.

           Theano dimension ordering:     (batches, channels, height, width)
           Tensorflow dimension ordering: (batches, height, width, channels)
        """

        output = super(MaxMinConvolution2D, self).call(x)

        if self.dim_ordering == 'th':
            output = K.concatenate([output, -output], axis=1)
        elif self.dim_ordering == 'tf':
            output = K.concatenate([output, -output], axis=3)

        output = self.activation_(output)
        return output

    def get_output_shape_for(self, input_shape):
        """The output shape is doubled along the axis representing channels due 
           to concatenation of two identical sized Convolution layers.
        """
        output_shape = super(MaxMinConvolution2D, self).get_output_shape_for(input_shape)

        output_shape = list(output_shape)
        if self.dim_ordering == 'th':
            output_shape[1] *= 2
        elif self.dim_ordering == 'tf':
            output_shape[3] *= 2

        return tuple(output_shape)
