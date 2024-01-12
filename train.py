import argparse
import ctypes


def main():
    parser = argparse.ArgumentParser(description='Training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')

    # Optional arguments
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--model_type', type=int, default=1, help='Model type (int) 1 - MultiresNet, 2 - Unet3++ (default: 1)')
    parser.add_argument('--warmup', type=float, default=1, help='Warmup epochs (float) (default: 1)')
    parser.add_argument('--pulse', type=float, default=9, help='Period in number of epochs to pulse learning rate (float) (default: 9)')
    parser.add_argument('--pulse_amplitude', type=float, default=10, help='Learning rate pulse amplitude (percentage) (default: 10)')

    args = parser.parse_args()

    # Access arguments using args.learning_rate, args.model_type, etc.
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Model Type: {args.model_type}")
    print(f"Warmup: {args.warmup}")
    print(f"Pulse: {args.pulse}")
    print(f"Pulse Amplitude: {args.pulse_amplitude}")

    # Rest of your training script...

if __name__ == "__main__":
    main()

