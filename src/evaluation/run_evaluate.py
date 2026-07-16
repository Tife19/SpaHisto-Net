
"""
Thin runner script for evaluating the model on the test set.
Designed for easy use inside Docker or from the command line.
"""

import argparse
from src.evaluation.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpaHisto-Net on Test Set")

    parser.add_argument("--predictions_dir", type=str, required=True,
                        help="Path to directory containing *_predictions.pt files")
    parser.add_argument("--label_dir", type=str, required=True,
                        help="Path to directory containing ground truth .h5ad files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (optional)")
    parser.add_argument("--save_results", action="store_true",
                        help="Whether to save gene-level correlations CSV")

    args = parser.parse_args()

    evaluate_model(
        predictions_dir=args.predictions_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        save_results=args.save_results
    )


if __name__ == "__main__":
    main()

