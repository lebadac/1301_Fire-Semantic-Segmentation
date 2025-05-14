import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import time

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

class FastAttentionLayer(layers.Layer):
    """
    Fast attention layer with residual connections.
    """
    def __init__(self, output_dim):
        super(FastAttentionLayer, self).__init__()
        self.output_dim = output_dim
        self.query_proj = layers.Dense(output_dim)
        self.key_proj = layers.Dense(output_dim)
        self.value_proj = layers.Dense(output_dim)

    def call(self, inputs):
        input_rank = inputs.shape.rank
        if input_rank == 4:
            b, h, w, c = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
            n = h * w
            x = tf.reshape(inputs, [b, n, c])
            Q = self.query_proj(tf.nn.l2_normalize(x, axis=-1))
            K = self.key_proj(tf.nn.l2_normalize(x, axis=-1))
            V = self.value_proj(x)
            KV = tf.matmul(K, V, transpose_a=True)
            Y = tf.matmul(Q, KV) / tf.cast(n, tf.float32)
            return tf.reshape(Y + x, [b, h, w, self.output_dim])
        elif input_rank == 3:
            n = tf.shape(inputs)[1]
            Q = self.query_proj(tf.nn.l2_normalize(inputs, axis=-1))
            K = self.key_proj(tf.nn.l2_normalize(inputs, axis=-1))
            V = self.value_proj(inputs)
            KV = tf.matmul(K, V, transpose_a=True)
            Y = tf.matmul(Q, KV) / tf.cast(n, tf.float32)
            return Y + inputs
        else:
            raise ValueError("Unsupported input rank.")

def tokenized_kan_block_student(inputs, token_dim, kan_layers=2):
    """
    Implements the tokenized KAN block for the student model.
    """
    tokens = layers.Reshape((-1, inputs.shape[-1]))(inputs)
    tokens = layers.Dense(token_dim, activation='relu')(tokens)

    x = tokens
    for _ in range(kan_layers):
        y = KANLayer(token_dim, token_dim)(x)
        y = layers.LayerNormalization()(y)
        x = layers.Add()([x, y])

    x = FastAttentionLayer(token_dim)(x)
    x = layers.LayerNormalization()(x)

    projected = layers.Conv2D(token_dim, (1, 1), padding='same', activation='relu')(inputs)
    x_reshaped = layers.Lambda(lambda x: tf.reshape(x, (-1, projected.shape[1], projected.shape[2], token_dim)))(x)
    tokens = layers.Add()([x_reshaped, projected])

    out = KANLayer(token_dim, token_dim)(tokens)
    out = layers.LayerNormalization()(out)

    return layers.Reshape((inputs.shape[1], inputs.shape[2], token_dim))(out)

def fuse_up(skip, up_input, out_channels):
    """
    Fuses skip connection with upsampled input.
    """
    upsampled = layers.UpSampling2D((2, 2), interpolation='bilinear')(up_input)
    height, width = upsampled.shape[1], upsampled.shape[2]
    skip_resized = layers.Resizing(height, width, interpolation='bilinear')(skip)

    if skip_resized.shape[-1] != upsampled.shape[-1]:
        skip_resized = layers.Conv2D(upsampled.shape[-1], (1, 1), padding='same', use_bias=False)(skip_resized)

    x = layers.Add()([upsampled, skip_resized])
    x = layers.ReLU()(x)
    x = layers.Conv2D(out_channels, (3, 3), padding='same', use_bias=False)(x)
    return x

def build_student_model(input_shape, kan_dim=16, num_kan_layers=2):
    """
    Builds the student model with MobileNetV2 encoder and lightweight KAN block.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        kan_dim (int): Dimension of the tokenized KAN block.
        num_kan_layers (int): Number of KAN layers in the block.

    Returns:
        Model: Student model.
    """
    inputs = layers.Input(shape=input_shape)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    for layer in base_model.layers:
        if 'block_13' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    c1 = base_model.get_layer('block_1_expand_relu').output
    c2 = base_model.get_layer('block_3_expand_relu').output
    c3 = base_model.get_layer('block_6_expand_relu').output
    c4 = base_model.get_layer('block_13_expand_relu').output

    bottleneck = tokenized_kan_block_student(c4, kan_dim, num_kan_layers)

    c4_skip = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(c4)
    c4_skip = FastAttentionLayer(64)(c4_skip)
    u1 = fuse_up(c4_skip, bottleneck, 32)

    c3_skip = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(c3)
    c3_skip = FastAttentionLayer(64)(c3_skip)
    u2 = fuse_up(c3_skip, u1, 16)

    c2_skip = layers.Conv2D(32, (1, 1), padding='same', use_bias=False)(c2)
    c2_skip = FastAttentionLayer(32)(c2_skip)
    u3 = fuse_up(c2_skip, u2, 8)

    c1_skip = layers.Conv2D(16, (1, 1), padding='same', use_bias=False)(c1)
    c1_skip = FastAttentionLayer(16)(c1_skip)
    u4 = fuse_up(c1_skip, u3, 8)

    u4 = layers.Dropout(0.05)(u4)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u4)

    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

def distillation_loss(y_true, y_pred, teacher_pred, temperature=4.0, alpha=0.5):
    """
    Computes the distillation loss combining hard and soft losses.

    Args:
        y_true: Ground truth labels.
        y_pred: Student model predictions.
        teacher_pred: Teacher model predictions.
        temperature (float): Temperature for softening probabilities.
        alpha (float): Weight for hard loss.

    Returns:
        float: Combined loss value.
    """
    hard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    hard_loss = tf.reduce_mean(hard_loss)

    student_soft = tf.nn.softmax(y_pred / temperature)
    teacher_soft = tf.nn.softmax(teacher_pred / temperature)
    soft_loss = tf.keras.losses.KLDivergence()(teacher_soft, student_soft)
    soft_loss = soft_loss * (temperature ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss

def train_student(teacher_model, X_train, Y_train, input_shape, model_dir="saved_weights", epochs=20, batch_size=8):
    """
    Trains the student model using knowledge distillation.

    Args:
        teacher_model: Pretrained teacher model.
        X_train (list): List of training video image arrays.
        Y_train (list): List of training video mask arrays.
        input_shape (tuple): Input shape for the model.
        model_dir (str): Directory to save model weights.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        str: Path to the saved student model weights.
    """
    teacher_model.trainable = False
    student_model = build_student_model(input_shape, kan_dim=16, num_kan_layers=2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            student_pred = student_model(x_batch, training=True)
            teacher_pred = teacher_model(x_batch, training=False)
            loss = distillation_loss(y_batch, student_pred, teacher_pred, temperature=4.0, alpha=0.5)
        gradients = tape.gradient(loss, student_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
        return loss

    X_train_all = np.concatenate(X_train, axis=0)
    Y_train_all = np.concatenate(Y_train, axis=0)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_all, Y_train_all))
    train_dataset = train_dataset.shuffle(buffer_size=500).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")
        epoch_start = time.time()
        step_losses = []

        pbar = tqdm(enumerate(train_dataset), total=len(train_dataset))
        for step, (x_batch, y_batch) in pbar:
            loss = train_step(x_batch, y_batch)
            step_losses.append(loss.numpy())
            pbar.set_description(f"Step {step} - Loss {loss.numpy():.6f}")

        epoch_loss = sum(step_losses) / len(step_losses)
        epoch_time = time.time() - epoch_start
        print(f"✅ Epoch {epoch+1} finished - Average Loss: {epoch_loss:.6f} - Time: {epoch_time:.2f}s")

    student_weights_path = os.path.join(model_dir, "distilled_student_model_weights.weights.h5")
    student_model.save_weights(student_weights_path)
    print(f"💾 Student model weights saved: {student_weights_path}")

    return student_weights_path, student_model
