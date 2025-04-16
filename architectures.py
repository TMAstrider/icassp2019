
from keras.layers import Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation
from keras.models import Model
from keras.regularizers import l2

# =====================================================================================


def get_model_baseline(params_learn=None, params_extract=None):
    """
    构建一个基础的CNN模型架构
    
    :param params_learn: 学习相关参数(如分类数)
    :param params_extract: 特征提取参数(如频谱图长度和Mel带数)
    :return: 编译好的Keras模型
    """
    # 定义输入形状 (1通道, 时间帧数, Mel带数)
    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    channel_axis = 1  # 通道轴位置(通道优先)
    n_class = params_learn.get('n_classes')  # 分类数量

    # 输入层
    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # 第一层卷积块
    spec_x = BatchNormalization(axis=1)(spec_x)  # 批标准化
    spec_x = Activation('relu')(spec_x)  # ReLU激活
    spec_x = Conv2D(24, (5, 5),  # 24个5x5卷积核
                   padding='same',  # 保持尺寸不变
                   kernel_initializer='he_normal',  # He初始化
                   data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)

    # 第二层卷积块(结构与第一层类似，但使用48个卷积核)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)

    # 第三层卷积块(结构与前两层类似)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    # 全连接部分
    spec_x = Flatten()(spec_x)  # 展平
    spec_x = Dropout(0.5)(spec_x)  # 50%的Dropout
    spec_x = Dense(64,  # 64个神经元的全连接层
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-3),  # L2正则化
                  activation='relu',
                  name='dense_1')(spec_x)

    # 输出层
    spec_x = Dropout(0.5)(spec_x)
    out = Dense(n_class,  # 输出神经元数=分类数
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-3),
               activation='softmax',  # 多分类softmax
               name='prediction')(spec_x)

    # 构建并返回模型
    model = Model(inputs=spec_start, outputs=out)
    return model
