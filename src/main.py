import os
from data_loader import load_video_data, get_dataset_splits
from unet import unet_model
from u_kan import unet_kan
from u_kan_lstm import unet_kan_lstm
from u_kan_lstm_mobilenetv2 import unet_kan_lstm_mobilenetv2
from evaluate import evaluate_model
from knowledge_distiilation import train_student, build_student_model

def main():
    # Configuration
    dataset_dir = "./data/Chino"
    model_dir = "./saved_weights"
    input_shape = (240, 240, 3)

    # Load test dataset
    video_sets = get_dataset_splits(dataset_dir)
    X_test, Y_test = load_video_data(dataset_dir, video_sets["test"], "test")
    print(f"Test: {len(X_test)} videos")

    # Evaluate U-Net model
    print("\nEvaluating U-Net model...")
    unet_model_instance = unet_model(input_shape)
    unet_model_instance.build(input_shape=(None, 240, 240, 3))
    unet_weights_path = os.path.join(model_dir, "unet.weights.h5")
    unet_model_instance.load_weights(unet_weights_path)
    unet_metrics = evaluate_model(unet_model_instance, X_test, Y_test, input_shape, "U-Net")

    # Evaluate U-Net-KAN model
    print("\nEvaluating U-Net-KAN model...")
    unet_kan_instance = unet_kan(input_shape)
    unet_kan_instance.build(input_shape=(None, 240, 240, 3))
    unet_kan_weights_path = os.path.join(model_dir, "u_kan.weights.h5")
    unet_kan_instance.load_weights(unet_kan_weights_path)
    unet_kan_metrics = evaluate_model(unet_kan_instance, X_test, Y_test, input_shape, "U-Net-KAN")

    # Evaluate U-Net-KAN-LSTM model
    print("\nEvaluating U-Net-KAN-LSTM model...")
    unet_kan_lstm_instance = unet_kan_lstm(input_shape)
    unet_kan_lstm_instance.build(input_shape=(None, 240, 240, 3))
    unet_kan_lstm_weights_path = os.path.join(model_dir, "unet_kan_lstm_weights_fold_1.weights.h5")
    unet_kan_lstm_instance.load_weights(unet_kan_lstm_weights_path)
    unet_kan_lstm_metrics = evaluate_model(unet_kan_lstm_instance, X_test, Y_test, input_shape, "U-Net-KAN-LSTM")

    # Evaluate U-Net-KAN-LSTM-MobileNetV2 model
    print("\nEvaluating U-Net-KAN-LSTM-MobileNetV2 model...")
    unet_kan_lstm_mobilenetv2_instance = unet_kan_lstm_mobilenetv2(input_shape)
    unet_kan_lstm_mobilenetv2_instance.build(input_shape=(None, 240, 240, 3))
    unet_kan_lstm_mobilenetv2_weights_path = os.path.join(model_dir, "unet_kan_lstm_mobilenetv2_weights_fold_1.weights.h5")
    unet_kan_lstm_mobilenetv2_instance.load_weights(unet_kan_lstm_mobilenetv2_weights_path)
    unet_kan_lstm_mobilenetv2_metrics = evaluate_model(unet_kan_lstm_mobilenetv2_instance, X_test, Y_test, input_shape, "U-Net-KAN-LSTM-MobileNetV2")

    # Evaluate Student model
    print("\nEvaluating Student model...")
    student_model = build_student_model(input_shape, kan_dim=16, num_kan_layers=2)
    student_model.build(input_shape=(None, 240, 240, 3))
    student_weights_path = os.path.join(model_dir, "distilled_student_model_weights.weights.h5")
    student_model.load_weights(student_weights_path)
    student_metrics = evaluate_model(student_model, X_test, Y_test, input_shape, "Student")

if __name__ == "__main__":
    main()