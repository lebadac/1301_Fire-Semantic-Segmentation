import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CallbackList
from sklearn.model_selection import KFold
from unet import unet_model
from u_kan import unet_kan
from u_kan_lstm import unet_kan_lstm
from u_kan_lstm_mobilenetv2 import unet_kan_lstm_mobilenetv2
from utils import dice_loss, bce_dice_loss

# --- Reproducibility ---
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Parameters ---
INITIAL_LR = 0.001
MIN_LR = 1e-6
BETA_1, BETA_2 = 0.9, 0.999
MAX_EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15
K_FOLDS = 5
MODEL_DIR = "../saved_weights"
os.makedirs(MODEL_DIR, exist_ok=True)

def create_model(model_name, input_shape, kan_dim=256, num_kan_layers=2):
    if model_name == 'unet':
        return unet_model(input_shape)
    elif model_name == 'u_kan':
        return unet_kan(input_shape)
    elif model_name == 'u_kan_lstm':
        return unet_kan_lstm(input_shape)
    elif model_name == 'u_kan_lstm_mobilenetv2':
        return unet_kan_lstm_mobilenetv2(input_shape)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

# --- Training Loop with KFold ---
def train_model(X_train, Y_train, input_shape):
    fold_results = []
    best_model_path = None
    best_val_loss = float('inf')

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n🔁 Training Fold {fold+1}/{K_FOLDS}")

        model_name = 'u_kan_lstm_mobilenetv2'  # Replace with any model you have
        model = create_model(model_name, input_shape)

        # Prepare fold data
        X_train_fold = [X_train[i] for i in train_idx]
        Y_train_fold = [Y_train[i] for i in train_idx]
        X_val_fold = [X_train[i] for i in val_idx]
        Y_val_fold = [Y_train[i] for i in val_idx]

        optimizer = Adam(learning_rate=INITIAL_LR, beta_1=BETA_1, beta_2=BETA_2)

        model.compile(
            optimizer=optimizer,
            loss=bce_dice_loss,
            metrics=[BinaryIoU(threshold=0.5, name='Binary_IoU')]
        )

        # Callbacks
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=MIN_LR, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)

        callbacks = [early_stop, reduce_lr]
        callback_list = CallbackList(callbacks, model=model)
        callback_list.set_model(model)
        callback_list.set_params({
            'epochs': MAX_EPOCHS,
            'verbose': 1,
            'do_validation': True,
            'metrics': ['val_loss']
        })
        callback_list.on_train_begin()

        # Training loop per epoch
        for epoch in range(MAX_EPOCHS):
            print(f"\n📅 [Fold {fold+1}] Epoch {epoch+1}/{MAX_EPOCHS}")
            callback_list.on_epoch_begin(epoch)

            video_indices = list(range(len(X_train_fold)))
            random.shuffle(video_indices)

            for vid_idx in video_indices:
                print(f"🎞️  Training on Video {vid_idx+1}/{len(X_train_fold)}")
                X_vid, Y_vid = X_train_fold[vid_idx], Y_train_fold[vid_idx]
                dataset_train = (
                    tf.data.Dataset.from_tensor_slices((X_vid, Y_vid))
                    .shuffle(buffer_size=len(X_vid))
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE)
                )
                model.fit(dataset_train, epochs=1, verbose=1)

            # Validation
            val_loss, val_iou = 0, 0
            for X_vid, Y_vid in zip(X_val_fold, Y_val_fold):
                dataset_val = tf.data.Dataset.from_tensor_slices((X_vid, Y_vid)).batch(1).prefetch(tf.data.AUTOTUNE)
                loss, iou = model.evaluate(dataset_val, verbose=0)
                val_loss += loss
                val_iou += iou
            val_loss /= len(X_val_fold)
            val_iou /= len(X_val_fold)

            print(f"✅ Validation Loss: {val_loss:.4f} - IoU: {val_iou:.4f}")

            # ✅ Current Learning Rate
            lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
            print(f"📉 Current Learning Rate: {lr:.6f}")

            # Callback step
            logs = {'val_loss': val_loss}
            callback_list.on_epoch_end(epoch, logs)

            if early_stop.stopped_epoch > 0:
                print("🛑 Early stopping triggered!")
                break

        callback_list.on_train_end()

        model_weights_path = os.path.join(MODEL_DIR, f"weights_fold_{fold+1}.weights.h5")
        model.save_weights(model_weights_path)
        print(f"💾 Model weights saved: {model_weights_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = model_weights_path

        fold_results.append({
            'fold': fold+1,
            'val_loss': val_loss,
            'val_iou': val_iou
        })

    # --- Summary ---
    avg_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_iou = np.mean([r['val_iou'] for r in fold_results])
    print("\n🏁 Training Completed!")
    print(f"📉 Average Validation Loss: {avg_loss:.4f}")
    print(f"📈 Average Validation IoU:  {avg_iou:.4f}")
    print(f"🌟 Best model weights: {best_model_path}")

    return best_model_path
