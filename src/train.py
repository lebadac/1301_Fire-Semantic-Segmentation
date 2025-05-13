import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import KFold
from utils import bce_dice_loss, preprocess_data

def train_model(model_fn, X_train, Y_train, input_shape, model_name, model_dir="saved_weights", k_folds=5, max_epochs=100, batch_size=32, seed=70):
    """
    Trains a model using K-fold cross-validation.

    Args:
        model_fn (callable): Function to create the model.
        X_train (list): List of training video image arrays.
        Y_train (list): List of training video mask arrays.
        input_shape (tuple): Input shape for the model.
        model_name (str): Name of the model for saving weights.
        model_dir (str): Directory to save model weights.
        k_folds (int): Number of folds for cross-validation.
        max_epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (fold_results, best_model_path)
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    initial_lr = 0.001
    min_lr = 1e-6
    beta_1, beta_2 = 0.9, 0.999
    patience = 10
    os.makedirs(model_dir, exist_ok=True)

    fold_results = []
    best_model_path = None
    best_val_loss = float('inf')

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n🔁 Training Fold {fold+1}/{k_folds} for {model_name}")
        model = model_fn(input_shape)

        X_train_fold = [X_train[i] for i in train_idx]
        Y_train_fold = [Y_train[i] for i in train_idx]
        X_val_fold = [X_train[i] for i in val_idx]
        Y_val_fold = [Y_train[i] for i in val_idx]

        print("Preprocessing training data...")
        X_train_fold_resized = []
        Y_train_fold_resized = []
        for x, y in zip(X_train_fold, Y_train_fold):
            x_r, y_r = preprocess_data(x, y, target_size=input_shape[:2])
            X_train_fold_resized.append(x_r)
            Y_train_fold_resized.append(y_r)

        print("Preprocessing validation data...")
        X_val_fold_resized = []
        Y_val_fold_resized = []
        for x, y in zip(X_val_fold, Y_val_fold):
            x_r, y_r = preprocess_data(x, y, target_size=input_shape[:2])
            X_val_fold_resized.append(x_r)
            Y_val_fold_resized.append(y_r)

        optimizer = Adam(learning_rate=initial_lr, beta_1=beta_1, beta_2=beta_2)
        model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[BinaryIoU(threshold=0.5, name='Binary_IoU')])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=min_lr, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
        early_stop.set_model(model)
        reduce_lr.set_model(model)

        for epoch in range(max_epochs):
            print(f"\n📅 [Fold {fold+1}] Epoch {epoch+1}/{max_epochs}")
            video_indices = list(range(len(X_train_fold_resized)))
            random.shuffle(video_indices)

            for vid_idx in video_indices:
                print(f"🎞️  Training on Video {vid_idx+1}/{len(X_train_fold_resized)}")
                X_vid, Y_vid = X_train_fold_resized[vid_idx], Y_train_fold_resized[vid_idx]
                dataset_train = (
                    tf.data.Dataset.from_tensor_slices((X_vid, Y_vid))
                    .shuffle(buffer_size=len(X_vid))
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
                )
                model.fit(dataset_train, epochs=1, verbose=1)

            val_loss, val_iou = 0, 0
            for X_vid, Y_vid in zip(X_val_fold_resized, Y_val_fold_resized):
                dataset_val = tf.data.Dataset.from_tensor_slices((X_vid, Y_vid)).batch(1).prefetch(tf.data.AUTOTUNE)
                loss, iou = model.evaluate(dataset_val, verbose=0)
                val_loss += loss
                val_iou += iou
            val_loss /= len(X_val_fold_resized)
            val_iou /= len(X_val_fold_resized)

            print(f"✅ Validation Loss: {val_loss:.4f} - IoU: {val_iou:.4f}")
            print(f"📉 Current Learning Rate: {float(tf.keras.backend.get_value(model.optimizer.learning_rate)):.6f}")

            early_stop.on_epoch_end(epoch, logs={'val_loss': val_loss})
            reduce_lr.on_epoch_end(epoch, logs={'val_loss': val_loss})

            if early_stop.stopped_epoch > 0:
                print("🛑 Early stopping triggered!")
                break

        model_weights_path = os.path.join(model_dir, f"{model_name}_weights_fold_{fold+1}.weights.h5")
        model.save_weights(model_weights_path)
        print(f"💾 Model weights saved: {model_weights_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = model_weights_path

        fold_results.append({'fold': fold+1, 'val_loss': val_loss, 'val_iou': val_iou})

    avg_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_iou = np.mean([r['val_iou'] for r in fold_results])
    print(f"\n🏁 Training Completed for {model_name}!")
    print(f"📉 Average Validation Loss: {avg_loss:.4f}")
    print(f"📈 Average Validation IoU:  {avg_iou:.4f}")
    print(f"🌟 Best model weights: {best_model_path}")

    return fold_results, best_model_path