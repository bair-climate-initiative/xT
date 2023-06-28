import json

DEFAULTS = {
    "network": "dpn",
    "encoder": "dpn92",
    "model_params": {},
    "optimizer": {
        "batch_size": 32,
        "type": "SGD",  # supported: SGD, Adam
        "momentum": 0.9,
        "weight_decay": 0,
        "clip": 1.,
        "learning_rate": 0.1,
        "classifier_lr": -1,
        "nesterov": True,
        "schedule": {
            "type": "constant",  # supported: constant, step, multistep, exponential, linear, poly
            "mode": "epoch",  # supported: epoch, step
            "epochs": 10,
            "params": {}
        }
    },
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}


def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_file, defaults=DEFAULTS,args=None):
    with open(config_file, "r") as fd:
        config = json.load(fd)
    _merge(defaults, config)
    if args is not None:
        if args.crop_size is not None:
            config["crop_size"] = args.crop_size
        if args.epoch is not None:
            config['optimizer']['schedule']['epochs'] = args.epoch
        if args.lr is not None:
            config['optimizer']['learning_rate'] = args.lr
        if args.weight_decay is not None:
            config['optimizer']['weight_decay'] = args.weight_decay
        if args.bs is not None:
            config['optimizer']['train_bs'] = args.bs
        if args.drop_path is not None:
            config['encoder_params']['drop_path_rate'] = args.drop_path
        if args.pretrained:
            pretrained = args.pretrained 
            if pretrained.lower() == 'true':
                pretrained = True
                print("Setting pretrained to True (Bool)")
            elif pretrained.lower() == 'false':
                pretrained = False
                print("Setting pretrained to False (Bool)")
            elif pretrained.lower() == 'default':
                print("Pretrained config is not changed, using config")
            else:
                print(f"Setting pretrained to {pretrained} (str)")
                config['encoder_params']['pretrained'] = pretrained
    return config
