from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Input, Multiply, Add

def DRCN(recursive_depth, input_channels, filter_num = 256): 
    """
    recursive_depth : numbers of recursive_conv2d.
    input_channels : channels of input_img.(gray → 1, RGB → 3)
    filter_num : filter numbers.(default 256)
    """

    Inferencd_conv2d = Conv2D(filters = filter_num, kernel_size = (3, 3), padding = "same", activation = "relu")

    """
    Inferencd_conv2d : Inference net.
    """
    #model
    input_shape = Input((None, None, input_channels))
    #Embedding net.
    conv2d_0 = Conv2D(filters = filter_num, kernel_size = (3, 3), padding = "same", activation = "relu")(input_shape)
    conv2d_1 = Conv2D(filters = filter_num, kernel_size = (3, 3), padding = "same", activation = "relu")(conv2d_0)                       
    #Inference net and Reconstruction net.
    weight_list = recursive_depth * [None]
    pred_list = recursive_depth * [None]

    for i in range(recursive_depth):
        Inferencd_output = Inferencd_conv2d(conv2d_1)
        Recon_0 = Conv2D(filters = filter_num, kernel_size = (3, 3), padding = "same", activation = "relu")(Inferencd_output)
        Recon_1 = Conv2D(filters = input_channels, kernel_size = (3, 3), padding = "same", activation = "relu")(Recon_0)
        weight_list[i] = Recon_1
    
    for i in range(recursive_depth):
        skip_connection = Add()([weight_list[i], input_shape])
        pred = Multiply()([weight_list[i], skip_connection])

        pred_list[i] = pred

    pred = Add()(pred_list)

    model = Model(inputs = input_shape, outputs = pred)
    model.summary()

    return model

