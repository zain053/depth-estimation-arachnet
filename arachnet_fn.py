# models/arachnet_fn.py
"""
ArachNet functional implementation (NO classes).
- 5 input branches (PFMs), each a UNet-like subnet
- Sum branch outputs + learnable per-pixel bias
- Final linear output for depth (meters)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# -------- per-pixel bias layer (no kernel) --------
class PerPixelBias(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bias = None

    def build(self, input_shape):
        # input: (B, H, W, 1) -> create trainable bias map (H, W, 1)
        h, w, c = input_shape[1], input_shape[2], input_shape[3]
        self.bias = self.add_weight(
            name="bias_map",
            shape=(h, w, c),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        return x + self.bias

def unet_subnet_fn(x, l2=1e-4, subnet_act='linear'):
    """A small UNet-like encoder-decoder returning a 1-channel map."""
    # Encoder
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(x)
    x1 = layers.Conv2D(32, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(x1)
    p1 = layers.MaxPool2D(2)(x1)

    x2 = layers.Conv2D(64, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(p1)
    x2 = layers.Conv2D(64, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(x2)
    p2 = layers.MaxPool2D(2)(x2)

    x3 = layers.Conv2D(128, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(p2)
    x3 = layers.Conv2D(128, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(x3)
    p3 = layers.MaxPool2D(2)(x3)

    # Bottleneck
    b = layers.Conv2D(256, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(l2))(p3)
    b = layers.Conv2D(256, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(l2))(b)

    # Decoder
    u3 = layers.UpSampling2D(2, interpolation="bilinear")(b)
    u3 = layers.Concatenate()([u3, x3])
    u3 = layers.Conv2D(128, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(u3)
    u3 = layers.Conv2D(128, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(u3)

    u2 = layers.UpSampling2D(2, interpolation="bilinear")(u3)
    u2 = layers.Concatenate()([u2, x2])
    u2 = layers.Conv2D(64, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(u2)
    u2 = layers.Conv2D(64, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(u2)

    u1 = layers.UpSampling2D(2, interpolation="bilinear")(u2)
    u1 = layers.Concatenate()([u1, x1])
    u1 = layers.Conv2D(32, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(u1)
    u1 = layers.Conv2D(32, 3, padding="same", activation="relu",
                       kernel_regularizer=regularizers.l2(l2))(u1)

    out = layers.Conv2D(1, 1, padding="same", activation=subnet_act,
                        kernel_regularizer=regularizers.l2(l2))(u1)
    return out

def build_arachnet(input_size=(480, 640, 1),   
                   features_num=5,
                   subnet_act='linear',
                   arach_act=None,
                   l2=1e-4):
    """
    Returns (main_model, full_model)
      - main_model: inputs=[5 PFMs], output=final depth (H,W,1)
      - full_model: outputs=[final, branch1..branch5]
    """
    inputs = [layers.Input(shape=input_size, name=f"pfm_{i+1}") for i in range(features_num)]
    subnet_outputs = [unet_subnet_fn(inp, l2=l2, subnet_act=subnet_act) for inp in inputs]
    summed = layers.Add(name="sum_subnets")(subnet_outputs)

    # Learnable per-pixel bias (no kernel)
    biased = PerPixelBias(name="per_pixel_bias")(summed)

    final_out = biased if (arach_act is None or arach_act == 'linear') \
               else layers.Activation(arach_act, name=f"arach_{arach_act}")(biased)

    main_model = models.Model(inputs=inputs, outputs=final_out, name="ArachNetFunctional")
    full_model = models.Model(inputs=inputs, outputs=[final_out] + subnet_outputs,
                              name="ArachNetFunctionalFull")
    return main_model, full_model
