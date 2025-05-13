import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Computes the Dice loss between true and predicted masks.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    """
    Combines binary cross-entropy and Dice loss.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def dice_coefficient(y_true, y_pred):
    """
    Computes the Dice coefficient between true and predicted masks.
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + 1e-6) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)

def preprocess_data(images, masks, target_size=(240, 240)):
    """
    Resizes images and masks to the target size.
    """
    resized_images = []
    resized_masks = []
    for img, mask in zip(images, masks):
        img = tf.cast(img, tffloat32)
        mask = tf.cast(mask, tf.float32)
        img_resized = tf.image.resize(img, target_size, method='bilinear')
        mask_resized = tf.image.resize(mask, target_size, method='nearest')
        resized_images.append(img_resized)
        resized_masks.append(mask_resized)
    return np.array(resized_images), np.array(resized_masks)