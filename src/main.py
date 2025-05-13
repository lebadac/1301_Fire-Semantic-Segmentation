import os
from data_loader import load_video_data, get_dataset_splits
from unet import unet_model
from u_kan import unet_kan
from u_kan_lstm import unet_kan_lstm
from u_kan_lstm_mobilenetv2 import unet_kan_lstm_mobilenetv2
from train import train_model
from evaluate import evaluate_model
from knowledge_distiilation import train_student, build_student_model

def main():
    # Configuration
    dataset_dir = "./data/Chino_v2"
    model_dir = "./saved_weights"
    input_shape = (240, 240, 3)

    # Load dataset
    video_sets = get_dataset_splits(dataset_dir)
    X_train, Y_train = load_video_data(dataset_dir, video_sets["train"], "train")
    X_test, Y_test = load_video_data(dataset_dir, video_sets["test"], "test")
    print(f"Train: {len(X_train)} videos")
    print(f"Test: {len(X_test)} videos")

    # Train U-Net model
    print("\nTraining U-Net model...")
    unet_results, unet_best_path = train_model(
        model_fn=unet_model,
        X_train=X_train,
        Y_train=Y_train,
        input_shape=input_shape,
        model_name="unet",
        model_dir=model_dir
    )

    # Train U-Net-KAN model
    print("\nTraining U-Net-KAN model...")
    unet_kan_results, unet_kan_best_path = train_model(
        model_fn=unet_kan,
        X_train=X_train,
        Y_train=Y_train,
        input_shape=input_shape,
        model_name="unet_kan",
        model_dir=model_dir
    )

    # Train U-Net-KAN-LSTM model
    print("\nTraining U-Net-KAN-LSTM model...")
    unet_kan_lstm_results, unet_kan_lstm_best_path = train_model(
        model_fn=unet_kan_lstm,
        X_train=X_train,
        Y_train=Y_train,
        input_shape=input_shape,
        model_name="unet_kan_lstm",
        model_dir=model_dir
    )

    # Train U-Net-KAN-LSTM-MobileNetV2 model (Teacher)
    print("\nTraining U-Net-KAN-LSTM-MobileNetV2 model...")
    unet_kan_lstm_mobilenetv2_results, unet_kan_lstm_mobilenetv2_best_path = train_model(
        model_fn=unet_kan_lstm_mobilenetv2,
        X_train=X_train,
        Y_train=Y_train,
        input_shape=input_shape,
        model_name="unet_kan_lstm_mobilenetv2",
        model_dir=model_dir
    )

    # Train Student model with distillation
    print("\nTraining Student model with distillation...")
    teacher_model = unet_kan_lstm_mobilenetv2(input_shape, kan_dim=256, num_kan_layers=2)
    teacher_model.build(input_shape=(None, 240, 240, 3))
    teacher_model.load_weights(unet_kan_lstm_mobilenetv2_best_path)
    student_weights_path, student_model = train_student(
        teacher_model=teacher_model,
        X_train=X_train,
        Y_train=Y_train,
        input_shape=input_shape,
        model_dir=model_dir
    )

    # Evaluate U-Net model
    print("\nEvaluating U-Net model...")
    unet_model_instance = unet_model(input_shape)
    unet_model_instance.build(input_shape=(None, 240, 240, 3))
    unet_model_instance.load_weights(unet_best_path)
    unet_metrics = evaluate_model(unet_model_instance, X_test, Y_test, input_shape, "U-Net")

    # Evaluate U-Net-KAN model
    print("\nEvaluating U-Net-KAN model...")
    unet_kan_instance = unet_kan(input_shape)
    unet_kan_instance.build(input_shape=(None, 240, 240, 3))
    unet_kan_instance.load_weights(unet_kan_best_path)
    unet_kan_metrics = evaluate_model(unet_kan_instance, X_test, Y_test, input_shape, "U-Net-KAN")

    # Evaluate U-Net-KAN-LSTM model
    print("\nEvaluating U-Net-KAN-LSTM model...")
    unet_kan_lstm_instance = unet_kan_lstm(input_shape)
    unet_kan_lstm_instance.build(input_shape=(None, 240, 240, 3))
    unet_kan_lstm_instance.load_weights(unet_kan_lstm_best_path)
    unet_kan_lstm_metrics = evaluate_model(unet_kan_lstm_instance, X_test, Y_test, input_shape, "U-Net-KAN-LSTM")

    # Evaluate U-Net-KAN-LSTM-MobileNetV2 model
    print("\nEvaluating U-Net-KAN-LSTM-MobileNetV2 model...")
    unet_kan_lstm_mobilenetv2_instance = unet_kan_lstm_mobilenetv2(input_shape)
    unet_kan_lstm_mobilenetv2_instance.build(input_shape=(None, 240, 240, 3))
    unet_kan_lstm_mobilenetv2_instance.load_weights(unet_kan_lstm_mobilenetv2_best_path)
    unet_kan_lstm_mobilenetv2_metrics = evaluate_model(unet_kan_lstm_mobilenetv2_instance, X_test, Y_test, input_shape, "U-Net-KAN-LSTM-MobileNetV2")

    # Evaluate Student model
    print("\nEvaluating Student model...")
    student_model.build(input_shape=(None, 240, 240, 3))
    student_model.load_weights(student_weights_path)
    student_metrics = evaluate_model(student_model, X_test, Y_test, input_shape, "Student")

if __name__ == "__main__":
    main()