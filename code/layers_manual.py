import math

import layers_keras
import numpy as np
import tensorflow as tf

BatchNormalization = layers_keras.BatchNormalization
Dropout = layers_keras.Dropout


class Conv2D(layers_keras.Conv2D):
    """
    Manually applies filters using the appropriate filter size and stride size
    """

    def call(self, inputs, training=False):
        ## If it's training, revert to layers implementation since this can be non-differentiable
        if training:
            return super().call(inputs, training)

        ## Otherwise, manually compute convolution at inference.
        ## Doesn't have to be differentiable. YAY!
        bn, h_in, w_in, c_in = inputs.shape  ## Batch #, height, width, # channels in input
        c_out = self.filters                 ## channels in output
        fh, fw = self.kernel_size            ## filter height & width
        sh, sw = self.strides                ## filter stride

        # Cleaning padding input.

        if self.padding == "SAME":
        #      ph = (fh - 1) // 2
        #      pw = (fw - 1) // 2
        # elif self.padding == "VALID":
        #      ph, pw = 0, 0
        # else:
        #     raise AssertionError(f"Illegal padding type {self.padding}")
            if (h_in % sh == 0):
                pad_along_height = max(fh - sh, 0)
            else:
                pad_along_height = max(fh - (h_in % sh), 0)
            if (w_in % sw == 0):
                pad_along_width = max(fw - sw, 0)
            else:
                pad_along_width = max(fw - (w_in % sw), 0)
            
            pad_top = pad_along_height // 2
            pad_btm = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
        elif self.padding == "VALID":
            pad_top, pad_btm, pad_left, pad_right = 0,0,0,0
        else:
            raise AssertionError(f"Illegal padding type {self.padding}")

        ## TODO: Convolve filter from above with the inputs.
        ## Note: Depending on whether you used SAME or VALID padding,
        ## the input and output sizes may not be the same
        
        #out put shape

        # if self.padding=="VALID":
        #     h_out = int((h_in - fh) / sh + 1)
        #     w_out = int((w_in - fw) / sw + 1)
        
        # if self.padding =="SAME":
        inputs = np.pad(inputs, ((0, 0), (pad_top, pad_btm), (pad_left, pad_right), (0, 0)), 'constant', constant_values=((0, 0),(0,0),(0,0),(0,0)))
        h_out = int(((h_in - fh + (pad_top + pad_btm))/sh)+1)
        w_out = int(((w_in - fw + (pad_left + pad_right))/sw)+1)
        

        ## Pad input if necessary
        # Cleaning padding input
        array = np.zeros(shape = (bn, h_out, w_out, c_out))
        
        for c_o in range(c_out):
            for c_i in range(c_in):
                for w in range(w_out):
                    for h in range(h_out):
                        var = np.reshape(tf.Variable(self.kernel[:,:,c_i,c_o]),(fh,fw))
                        
                        cal = (inputs[:,(h*sh):(h*sh+fh),(w*sw):(w*sw+fw),c_i]*var)
                        
                        array[:,h,w,c_o] += cal.sum(axis=(1,2))        
                        
                        result = tf.convert_to_tensor(array, dtype=tf.float32)
        # for i in range(h_out):
        #     i_in = i*sh
        #     for j in range(w_out):
        #         j_in = j*sw
        #         in_key = inputs[:,i_in:i_in+fh,j_in:j_in+fw,:]

        #         in_key = np.expand_dims(in_key,axis=1)
                
        #         outputs[:,i,j,:] = (my_kernel*in_key).sum(axis=-1).sum(axis=-1).sum(axis=-1)
        
        # return tf.convert_to_tensor(outputs,dtype=tf.float32)

        return result
        



        ## Calculate correct output dimensions

        ## Iterate and apply convolution operator to each image

        ## PLEASE RETURN A TENSOR using tf.convert_to_tensor(your_array, dtype=tf.float32)
        #return None
