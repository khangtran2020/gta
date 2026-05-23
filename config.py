import argparse


def add_general_group(group):
    group.add_argument(
        "--mode",
        type=str,
        default="data",
        help="""Runing mode the program: data, train, test""",
    )
    group.add_argument(
        "--name",
        type=str,
        default="testing",
        help="Experiment name, used for logging, checkpointing, wandb, etc.",
    )
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--debug", action="store_true", help="debug mode")


def add_data_group(group):
    group.add_argument(
        "--data_path",
        type=str,
        help="path to model dict",
        default="./data/model_info.json",
    )
    group.add_argument(
        "--num_cpu",
        type=int,
        default=-1,
        help="number of cpus to use for parallel processing (-1 to use all available cores)",
    )


def add_model_group(group):

    group.add_argument(
        "--grad_transformer_model",
        type=str,
        help="Pretrained model to use for the gradient transformer",
        default="google/flan-t5-large",
    )
    group.add_argument(
        "--projection_model",
        type=str,
        help="Pretrained model to use for the projection of the input features",
        default="microsoft/resnet-18",
    )
    group.add_argument(
        "--input_dimension",
        type=int,
        help="the length and width of the input that we will pad the input features to",
        default=2048,
    )
    group.add_argument(
        "--dropout",
        type=float,
        help="dropout rate",
        default=0.2,
    )
    group.add_argument(
        "--max_num_checkpoint",
        type=int,
        help="maximum number of checkpoints",
        default=5,
    )
    group.add_argument(
        "--continue_training",
        action="store_true",
        help="continue training from checkpoint",
    )
    group.add_argument(
        "--num_samples_per_input",
        type=int,
        help="number of samples to sample for each input by adding gaussian noise to the model weights",
        default=10,
    )


def add_training_group(group):

    group.add_argument(
        "--max_seq_length",
        type=int,
        help="max sequence length",
        default=512,
    )
    group.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
        default=1,
    )
    group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="gradient accumulation steps",
        default=16,
    )
    group.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="use DeepSpeed for training optimization (ZeRO-2 for 20-40%% speedup, ZeRO-3 for maximum memory efficiency)",
    )
    group.add_argument(
        "--deepspeed_config",
        type=str,
        help="path to DeepSpeed config file (supports ZeRO-2 and ZeRO-3)",
        default="configs/deepspeed_zero2.json",
    )
    group.add_argument(
        "--learning_rate",
        type=float,
        help="learning rate",
        default=1e-3,
    )
    group.add_argument(
        "--rope_theta",
        type=float,
        help="rope theta",
        default=500000.0,
    )
    group.add_argument(
        "--max_grad_norm",
        type=float,
        help="max gradient norm",
        default=1.0,
    )
    group.add_argument(
        "--num_train_epochs",
        type=int,
        help="number of training epochs",
        default=3,
    )
    group.add_argument(
        "--dtype",
        type=str,
        help="model and data type",
        default="bf16",
    )
    group.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="resume from checkpoint",
    )
    group.add_argument(
        "--logging_steps",
        type=int,
        help="number of steps to logs",
        default=128,
    )
    group.add_argument(
        "--save_steps",
        type=int,
        help="number of steps to save checkpoint",
        default=200,
    )
    group.add_argument(
        "--validating_steps",
        type=int,
        help="number of steps to validate",
        default=200,
    )


def add_testing_group(group):

    group.add_argument(
        "--model_path",
        type=str,
        help="path to the module to generate test cases",
        default=None,
    )
    group.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="maximum number of tokens to generate",
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # Add local_rank for DeepSpeed launcher compatibility
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by DeepSpeed launcher)",
    )

    general_group = parser.add_argument_group(title="General configuration")
    data_group = parser.add_argument_group(title="Data-related configuration")
    training_group = parser.add_argument_group(title="Training-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")
    testgen_group = parser.add_argument_group(
        title="Test-case generation configuration"
    )

    add_data_group(data_group)
    add_general_group(general_group)
    add_training_group(training_group)
    add_model_group(model_group)
    add_testing_group(testgen_group)

    return parser.parse_args()
