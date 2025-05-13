import numpy as np
import tensorflow as tf
import time
from utils import dice_coefficient, preprocess_data
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, Y_test, input_shape, model_name, batch_size=8):
    """
    Evaluates the model on the test set and computes performance metrics.

    Args:
        model (Model): Trained Keras model.
        X_test (list): List of test video image arrays.
        Y_test (list): List of test video mask arrays.
        input_shape (tuple): Input shape for the model.
        model_name (str): Name of the model for logging.
        batch_size (int): Batch size for prediction.

    Returns:
        dict: Metrics including pixel accuracy, mean IoU, mean Dice, FPS, etc.
    """
    Y_pred_list = []
    Y_test_resized_list = []
    start_time = time.time()

    for X_vid, Y_vid in zip(X_test, Y_test):
        X_vid = np.array(X_vid)
        Y_vid = np.array(Y_vid)
        X_vid_resized, Y_vid_resized = preprocess_data(X_vid, Y_vid, target_size=input_shape[:2])
        Y_pred_vid = model.predict(X_vid_resized, batch_size=batch_size, verbose=1)
        Y_pred_list.append(Y_pred_vid)
        Y_test_resized_list.append(Y_vid_resized)

    end_time = time.time()

    Y_pred = np.concatenate(Y_pred_list, axis=0)
    Y_test_resized = np.concatenate(Y_test_resized_list, axis=0)

    num_images = Y_pred.shape[0]
    average_time_per_image = (end_time - start_time) / num_images
    fps = 1.0 / average_time_per_image

    Y_pred = (Y_pred > 0.5).astype(np.uint8)

    num_classes = 2
    iou_scores = []
    dice_scores = []
    pixel_accuracy = 0
    mean_accuracy = 0
    freq_weighted_iou = 0
    total_pixels = np.prod(Y_test_resized.shape)

    for i in range(num_classes):
        y_true_class = (Y_test_resized == i).astype(np.uint8)
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
    mean_dice = np.mean(dice_scores)

    metrics = {
        'pixel_accuracy': pixel_accuracy,
        'mean_accuracy': mean_accuracy,
        'mean_iou': mean_iou,
        'freq_weighted_iou': freq_weighted_iou,
        'mean_dice': mean_dice,
        'average_time_per_image': average_time_per_image,
        'fps': fps
    }

    print(f"\n📊 Evaluation Results for {model_name}:")
    print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Frequency Weighted IoU: {freq_weighted_iou:.4f}")
    print(f"Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"Average time per image: {average_time_per_image:.6f} seconds")
    print(f"Frames Per Second (FPS): {fps:.2f}")

    # Visualize predictions for the first video
    X_vid = np.array(X_test[0])
    Y_vid = np.array(Y_test[0])
    Y_pred_vid = Y_pred_list[0]
    for i in range(min(5, len(X_vid))):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(X_vid[i])
        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(Y_vid[i].squeeze(), cmap='gray')
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(Y_pred_vid[i].squeeze(), cmap='gray')
        plt.savefig(f"{model_name}_prediction_{i}.png")
        plt.close()

    return metrics