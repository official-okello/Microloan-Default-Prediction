import argparse
from src.pipeline import load_data, feature_engineering, split_data, build_train_pipeline

def main(dataset_path):
    raw_data = load_data(dataset_path)
    X_train, X_test, y_train, y_test = split_data(raw_data)
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)
    build_train_pipeline(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a microloan default prediction model.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    args = parser.parse_args()
    main(args.data)

# to run:
# ```bash
# python cli.py --data ./Microloan Default Prediction/datasets/simulated_loan_data.csv