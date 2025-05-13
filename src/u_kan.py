import tensorflow as tf
from tensorflow.keras import layers, models

class KANLayer(layers.Layer):
    """
    Custom KAN layer implementing learnable activation functions.
    """
    def __init__(self, input_dim, output_dim, activation='gelu'):
        super(KANLayer, self).__init__()
        self.weight = self.add_weight(
            shape=(output_dim, input_dim),
            initializer="he_normal",
            trainable=True,
            name="kan_weights"
        )
        self.bias = self.add_weight(
            shape=(output_dim,),
            initializer="zeros",
            trainable=True,
            name="kan_bias"
        )
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        x = tf.tensordot(inputs, self.weight, axes=1) + self.bias
        return self.activation(x)

def tokenized_kan_block(inputs, token_dim, kan_layers=2):
    """
    Implements the Tok-KAN block by flattening inputs, applying KAN, and returning reshaped outputs.
    """
    tokens = layers.Reshape((-1, inputs.shape[-1]))(inputs)
    tokens = layers.Dense(token_dim, activation='relu')(tokens)
    for _ in range(kan_layers):
        processed_tokens = KANLayer(token_dim, token_dim)(tokens)
        processed_tokens = layers.LayerNormalization()(processed_tokens)
        tokens = layers.Add()([processed_tokens, tokens])
    return layers.Reshape((inputs.shape[1], inputs.shape[2], token_dim))(tokens)

def unet_kan(input_shape=(240, 240, 3), kan_dim=256, num_kan_layers=2):
    """
    Builds the U-Net-KAN model with a tokenized KAN block in the bottleneck.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        kan_dim (int): Dimension of the tokenized KAN block.
        num_kan_layers (int): Number of KAN layers in the block.

    Returns:
        Model: U-Net-KAN model.
    """
    inputs = layers.Input(shape=input_shape)

    # Contracting path
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck: Tokenized KAN Block
    kan_block_output = tokenized_kan_block(p4, kan_dim, num_kan_layers)

    # Expanding path
    u1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(kan_block_output)
    u1 = layers.concatenate([u1, c4])
    u1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u1)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(u1)
    u2 = layers.concatenate([u2, c3])
    u2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)

    u3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u2)
    u3 = layers.concatenate([u3, c2])
    u3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)

    u4 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u3)
    u4 = layers.concatenate([u4, c1])
    u4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u4)

    u4 = layers.Dropout(0.05)(u4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u4)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model