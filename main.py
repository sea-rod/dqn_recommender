# main.py
import argparse
from train.train_dqn import main as train_main
from train.evaluate import evaluate as eval_main


def main():
    parser = argparse.ArgumentParser(description="DQN-based Recommendation System")
    parser.add_argument(
        "--mode", choices=["train", "evaluate"], required=True, help="Mode to run"
    )
    args = parser.parse_args()

    if args.mode == "train":
        train_main()
    elif args.mode == "evaluate":
        eval_main("models/dqn_model.pth")


if __name__ == "__main__":
    main()
