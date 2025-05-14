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
    dataset_dir = "../data/firedataset"
    model_dir = "../model_weights"
    input_shape = (288, 288, 3)

    # Load test dataset
    video_sets = get_dataset_splits(dataset_dir)
    X_test, Y_test = load_video_data(dataset_dir, video_sets["test"], "test")
    print(f"Test: {len(X_test)} videos")

    while True:
        # User input for model evaluation
        print("\nSelect a model to evaluate:")
        print("1. U-Net")
        print("2. U-Net-KAN")
        print("3. U-Net-KAN-LSTM")
        print("4. U-Net-KAN-LSTM-MobileNetV2")
        print("5. Student model")
        print("0. Exit")

        choice = input("Enter the number of the model you want to evaluate (or 0 to exit): ")

        # Exit the loop if the user chooses option 6
        if choice == '0':
            print("Exiting the program.")
            break

        # Evaluate selected model
        if choice == '1':
            print("\nEvaluating U-Net model...")
            unet_model_instance = unet_model(input_shape)
            unet_model_instance.build(input_shape=(None, 240, 240, 3))
            unet_weights_path = os.path.join(model_dir, "unet.weights.h5")
            try:
                unet_model_instance.load_weights(unet_weights_path)
                print("Weights loaded successfully")
            except ValueError as e:
                print(f"Error loading weights: {e}")
            unet_metrics = evaluate_model(unet_model_instance, X_test, Y_test)

        # Evaluate U-Net-KAN model
        elif choice == '2':
            print("\nEvaluating U-Net-KAN model...")
            unet_kan_instance = unet_kan(input_shape)
            unet_kan_weights_path = os.path.join(model_dir, "u_kan.weights.h5")
            try:
                unet_kan_instance.load_weights(unet_kan_weights_path)
                print("Weights loaded successfully")
            except ValueError as e:
                print(f"Error loading weights: {e}")
            unet_kan_metrics = evaluate_model(unet_kan_instance, X_test, Y_test)

        # Evaluate U-Net-KAN-LSTM model
        elif choice == '3':
            print("\nEvaluating U-Net-KAN-LSTM model...")
            unet_kan_lstm_instance = unet_kan_lstm(input_shape)
            unet_kan_lstm_weights_path = os.path.join(model_dir, "u_kan_lstm.weights.h5")
            try:
                unet_kan_lstm_instance.load_weights(unet_kan_lstm_weights_path)
                print("Weights loaded successfully")
            except ValueError as e:
                print(f"Error loading weights: {e}")
            unet_kan_lstm_metrics = evaluate_model(unet_kan_lstm_instance, X_test, Y_test)

        # Evaluate U-Net-KAN-LSTM-MobileNetV2 model
        elif choice == '4':
            print("\nEvaluating U-Net-KAN-LSTM-MobileNetV2 model...")
            unet_kan_lstm_mobilenetv2_instance = unet_kan_lstm_mobilenetv2(input_shape)
            unet_kan_lstm_mobilenetv2_weights_path = os.path.join(model_dir, "u_kan_lstm_mobilenetv2.weights.h5")
            try:
                unet_kan_lstm_mobilenetv2_instance.load_weights(unet_kan_lstm_mobilenetv2_weights_path)
                print("Weights loaded successfully")
            except ValueError as e:
                print(f"Error loading weights: {e}")
            unet_kan_lstm_mobilenetv2_metrics = evaluate_model(unet_kan_lstm_mobilenetv2_instance, X_test, Y_test)

        # Evaluate Student model
        elif choice == '5':
            print("\nEvaluating Student model...")
            student_model = build_student_model(input_shape, kan_dim=16, num_kan_layers=2)
            student_weights_path = os.path.join(model_dir, "distilled_student_model_weights.weights.h5")
            try:
                student_model.load_weights(student_weights_path)
                print("Weights loaded successfully")
            except ValueError as e:
                print(f"Error loading weights: {e}")
            student_metrics = evaluate_model(student_model, X_test, Y_test)

        else:
            print("Invalid choice. Please select a number from 1 to 6.")


if __name__ == "__main__":
    main()
