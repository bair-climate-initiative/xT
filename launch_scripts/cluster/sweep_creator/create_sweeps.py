from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from itertools import product
from pathlib import Path

from omegaconf import OmegaConf


def deep_get(dictionary, keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )


def deep_set(dictionary, keys, value):
    dictionary_copy = deepcopy(dictionary)
    keys_list = keys.split(".")
    *nested_keys, last_key = keys_list

    # Use defaultdict instead of setdefault
    current_dict = reduce(
        lambda d, key: d[key],
        nested_keys,
        defaultdict(dict, dictionary_copy),
    )

    current_dict[last_key] = value
    dictionary_copy[nested_keys[0]] = deepcopy(current_dict)
    return dictionary_copy


def parse_args():
    parser = ArgumentParser("Create sweep files for Gigaformer experiments")
    parser.add_argument(
        "--sweep_options",
        default=None,
        type=Path,
        help="Path to the file that contains the sweep options",
    )
    parser.add_argument(
        "--base_config",
        default=None,
        type=Path,
        help="Path to config file upon which to base sweeps",
    )
    parser.add_argument(
        "--output_folder",
        default=None,
        type=Path,
        help="Path to folder where the sweeps will be saved",
    )
    parser.add_argument(
        "--experiment_base_name",
        default=None,
        type=str,
        help="Base name for the experiment name",
    )

    return parser.parse_args()


opt_key_map = {
    "optimizer.name": "optimizer",
    "optimizer.base_lr": "blr",
    "train.batch_size": "bs",
    "optimizer.warmup_epochs": "warmup",
}


def main():
    args = parse_args()
    sweep_options = OmegaConf.load(args.sweep_options)
    base_config = OmegaConf.load(args.base_config)
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    print(OmegaConf.to_yaml(sweep_options))

    combs = []
    for options in sweep_options["key_names"]:
        combs.append(list(product([options["name"]], options["options"])))
    all_combs = list(product(*combs))

    for combination in all_combs:
        modified_config = deepcopy(base_config)
        experiment_name = args.experiment_base_name
        for key, opt in combination:
            modified_config = deep_set(modified_config, key, opt)
            if key == "optimizer.name" and opt == "adamw":
                # Set AdamW default betas
                modified_config = deep_set(
                    modified_config, "optimizer.betas", [0.9, 0.999]
                )

            key_name = opt_key_map.get(key, key.split(".")[-1])
            key_name = key_name.replace("_", "-")
            if key_name == "blr":
                experiment_name += f"_{key_name}-{opt:.0e}"
            else:
                experiment_name += f"_{key_name}-{opt}"

            modified_config["name"] = experiment_name

        out_file = open(
            (args.output_folder / experiment_name).with_suffix(".yaml"), "w"
        )
        out_file.write(OmegaConf.to_yaml(modified_config))


if __name__ == "__main__":
    main()
