from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from itertools import product
from pathlib import Path

from omegaconf import OmegaConf


def deep_get(nested_dict, key_path):
    keys = key_path.split(".")
    return reduce(lambda d, key: d.get(key, {}), keys, nested_dict)


def deep_set(dictionary, key, nested_keys, i, value):
    if i == len(nested_keys) - 1:
        dictionary[nested_keys[i]] = value
        return
    else:
        deep_set(dictionary[nested_keys[i]], key, nested_keys, i + 1, value)


def parse_args():
    parser = ArgumentParser("Create sweep files for Gigaformer experiments")
    parser.add_argument(
        "--sweep_options",
        "-s",
        default=None,
        type=Path,
        help="Path to the file that contains the sweep options",
    )
    parser.add_argument(
        "--base_config",
        "-b",
        default=None,
        type=Path,
        help="Path to config file upon which to base sweeps",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        default=None,
        type=Path,
        help="Path to folder where the sweeps will be saved",
    )
    parser.add_argument(
        "--experiment_base_name",
        "-e",
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
    "data.crop_size": "crop_size",
    "model.backbone.img_size": "img_size",
    "model.xl_context.enabled": "xl",
    "model.backbone_class": "backbone",
}

opt_map = {
    "swinv2_tiny_window16_256_timm_xview": "swinv2-tiny",
    "swinv2_small_window16_256_timm_xview": "swinv2-small",
    "swinv2_base_window16_256_timm_xview": "swinv2-base",
    "swinv2_large_window16_256_timm_xview": "swinv2-large",
}


def main():
    args = parse_args()
    import os
    print(os.getcwd())
    sweep_options = OmegaConf.load(args.sweep_options)
    base_config = OmegaConf.load(args.base_config)
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    print(OmegaConf.to_yaml(sweep_options))

    combs = []
    for options in sweep_options["key_names"]:
        if options.get("group", None) is not None:
            all_options = []
            for group_options in options["group"]:
                all_options.append(list(group_options.items()))
            combs.append(all_options)
        else:
            combs.append(list(product([options["name"]], options["options"])))
    all_combs = list(product(*combs))

    # Flatten the grouped options 
    all_combs_temp = []
    for comb in all_combs:
        temp = []
        for opt in comb:
            if isinstance(opt, list):
                temp += opt
            else:
                temp.append(opt)
        all_combs_temp.append(temp)
    all_combs = all_combs_temp

    for combination in all_combs:
        modified_config = deepcopy(base_config)
        experiment_name = args.experiment_base_name
        for key, opt in combination:
            deep_set(modified_config, key, key.split("."), 0, opt)
            if key == "optimizer.name" and opt == "adamw":
                # Set AdamW default betas
                deep_set(
                    modified_config, "optimizer.betas", key.split("."), 0, [0.9, 0.999]
                )

            key_name = opt_key_map.get(key, key.split(".")[-1])
            key_name = key_name.replace("_", "-")

            opt= opt_map.get(opt, opt)
            if key_name == "blr":
                experiment_name += f"_{key_name}-{opt:.0e}"
            else:
                experiment_name += f"_{key_name}-{opt}"

            modified_config["name"] = experiment_name

        if (
            deep_get(modified_config, "data.crop_size")
            % deep_get(modified_config, "model.backbone.img_size")
            != 0
        ):
            continue

        out_file = open(
            (args.output_folder / experiment_name).with_suffix(".yaml"), "w"
        )
        out_file.write(OmegaConf.to_yaml(modified_config))


if __name__ == "__main__":
    main()
