import argparse

from tiald_trainer.joint_labels_trainer import MODEL_TYPES, TiALDJointLabelsTrainer


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate a joint-tasks model on TiALD.")
    parser.add_argument(
        "command",
        choices=["train", "eval"],
        help="Command: `train` or `eval`.",
    )
    parser.add_argument(
        "--input_model_name",
        type=str,
        required=True,
        help="Name or path of base model used for training/evaluation",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=MODEL_TYPES,
        help=f"Type of model to use, one of {MODEL_TYPES}",
    )
    parser.add_argument(
        "--output_model_name",
        type=str,
        required=True,
        help="Name used to save the trained model and predictions file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="fgaim/tigrinya-abusive-language-detection",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Hugging Face token",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum length of input text in tokens",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per device for training/evaluation",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--curriculum_loss",
        action="store_true",
        help="Use curriculum loss",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--search_n_trials",
        type=int,
        default=10,
        help="Number of trials for hyperparameter search",
    )
    parser.add_argument(
        "--report_to_wandb",
        action="store_true",
        help="Report to Weights & Biases",
    )

    return parser.parse_args()


def main():
    args = parse_cli_args()

    trainer = TiALDJointLabelsTrainer(
        input_model_name=args.input_model_name,
        model_type=args.model_type,
        output_model_name=args.output_model_name,
        dataset_name=args.dataset_name,
        hf_token=args.hf_token,
        output_dir=args.output_dir,
        max_length=args.max_length,
        lr=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        epochs=args.epochs,
        curriculum_loss=args.curriculum_loss,
        report_to="wandb" if args.report_to_wandb else "none",
    )

    if args.command == "eval":
        trainer.evaluate()
    elif args.command == "train":
        trainer.train()
    else:
        raise ValueError(f"Invalid command: {args.command}! Please use `train` or `eval`.")


if __name__ == "__main__":
    main()
