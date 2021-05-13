import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Add, Average

def DRCN(recursive_depth, input_channels, filter_num = 256): 
    """
    recursive_depth : numbers of recursive_conv2d.
    input_channels : channels of input_img.(gray → 1, RGB → 3)
    filter_num : filter numbers.(default 256)
    """

    Inferencd_conv2d = Conv2D(filters = filter_num,
                            kernel_size = (3, 3),
                            padding = "same",
                            activation = "relu")

    Recon_0 = Conv2D(filters = filter_num,
                        kernel_size = (3, 3),
                        padding = "same",
                        activation = "relu")

    Recon_1 = Conv2D(filters = input_channels,
                        kernel_size = (3, 3),
                        padding = "same",
                        activation = "relu")
    """
    Inferencd_conv2d : Inference net.
    Recon_0, Recon_1 : Reconstruction net.
    """

    #model
    input_shape = Input((None, None, input_channels))
    
    #Embedding net.
    conv2d_0 = Conv2D(filters = filter_num,
                        kernel_size = (3, 3),
                        padding = "same",
                        activation = "relu"
                        )(input_shape)
    conv2d_1 = Conv2D(filters = filter_num,
                        kernel_size = (3, 3),
                        padding = "same",
                        activation = "relu"
                        )(conv2d_0)
                        
    #Inference net and Reconstruction net.
    H = []  

    for i in range(recursive_depth):
        Inferencd_output = Inferencd_conv2d(conv2d_1)
        Reconstruction_1 = Recon_0(Inferencd_output)
        Reconstruction_2 = Recon_1(Reconstruction_1)
        H.append(Reconstruction_2)

    final_output = Average()(H)
    skip_connection = Add()([input_shape, final_output])

    model = Model(inputs = input_shape, outputs = skip_connection)

    model.summary()

    return model

