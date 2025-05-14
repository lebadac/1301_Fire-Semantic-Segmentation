import numpy as np
import tensorflow as tf
import time
from utils import dice_coefficient


# Evaluate Model function
def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the model on the test set and computes performance metrics.

        Args:
            model (Model): Trained Keras model.
            X_test (list): List of test video image arrays.
            Y_test (list): List of test video mask arrays.

        Returns:
            dict: Metrics including pixel accuracy, mean IoU, mean Dice, FPS, etc.
    """
    # Warm-up the model with 32 images
    warmup_vid = np.array(X_test[0])
    _ = model.predict(warmup_vid[:32], batch_size=16)

    # Start measuring time
    Y_pred_list = []
    start_time = time.time()

    # Predict for each video in the test set
    for X_vid in X_test:
        X_vid = np.array(X_vid)  # Convert list of frames to NumPy array
        Y_pred_vid = model.predict(X_vid, batch_size=16, verbose=0)
        Y_pred_list.append(Y_pred_vid)

    end_time = time.time()

    # Combine all predictions into a NumPy array
    Y_pred = np.concatenate(Y_pred_list, axis=0)

    # Calculate number of images in the test set
    num_images = Y_pred.shape[0]
    average_time_per_image = (end_time - start_time) / num_images
    fps = 1.0 / average_time_per_image

    # Convert predictions to binary labels
    Y_pred = (Y_pred > 0.5).astype(np.uint8)

    # Initialize metrics
    num_classes = 2
    iou_scores = []
    dice_scores = []
    pixel_accuracy = 0
    mean_accuracy = 0
    freq_weighted_iou = 0
    total_pixels = sum(np.prod(np.array(Y_vid).shape) for Y_vid in Y_test)

    # Compute metrics for each class
    for i in range(num_classes):
        y_true_class = np.concatenate([(Y_vid == i).astype(np.uint8) for Y_vid in Y_test], axis=0)
        y_pred_class = (Y_pred == i).astype(np.uint8)

        intersection = np.sum(y_true_class * y_pred_class)
        union = np.sum(y_true_class) + np.sum(y_pred_class) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = dice_coefficient(y_true_class, y_pred_class)

        iou_scores.append(iou)
        dice_scores.append(dice)
        pixel_accuracy += intersection
        mean_accuracy += intersection / (np.sum(y_true_class) + 1e-6)
        freq_weighted_iou += (np.sum(y_true_class) / total_pixels) * iou

    pixel_accuracy /= total_pixels
    mean_accuracy /= num_classes
    mean_iou = np.mean(iou_scores)

    # Print results
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Frequency Weighted IoU: {freq_weighted_iou:.4f}")
    print(f"Average time per image: {average_time_per_image:.6f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")

    # Returning metrics in a dictionary for further use if needed
    metrics = {
        "pixel_accuracy": pixel_accuracy,
        "mean_accuracy": mean_accuracy,
        "mean_iou": mean_iou,
        "frequency_weighted_iou": freq_weighted_iou,
        "average_time_per_image": average_time_per_image,
        "fps": fps
    }

    return metrics
