import json
import logging
import sys
from functools import partial

import torch
from occ.run_model import run_model
from occ.utils import hook_attn_map, print_elements, seed_all
from occ.vit import ViTModel

# TODO
    # pick back up at run_model.py 133 -> work on vit.py and utils.py
    # custom dataset split
    # docstrings
    # logging
    # reproducibility check
    # readme
    # dinov3
    # better attention map output options / functionality

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the config file from the cmd line argument
    if len(sys.argv) > 1:
        cfg_string = sys.argv[1]
        cfg = open(cfg_string)
    else:
        raise ValueError(
            "Please provide an argument for a json config location. E.g. `python ViT.py ./data/config.json`"
        )

    cfg = json.load(cfg)

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'{cfg['general']['model_id']}_{cfg['general']['datetime']}.log',
                        encoding='utf-8',
                        level=logging.INFO # TODO cfg
                        )

    # apply a seed for reproducibility
    seed_all(cfg["general"]["seed"])

    if cfg["general"]["verbose"]:
        logger.info("Print config.json items")
        print_elements(cfg)
        logger.info("End config print")

    model = ViTModel(config=cfg)

    attn_maps = []
    if cfg["output"]["include_attention_overlay"]:
        partial_hook = partial(hook_attn_map, attn_maps=attn_maps)
        # register the hook to the last attention layer of OCC model
        model.model.blocks[11].attn.qkv.register_forward_hook(partial_hook)

    results = run_model(
        model=model,
        config=cfg,
        device=device,
        attn_maps=attn_maps,
        verbose=cfg["general"]["verbose"],
    )


if __name__ == "__main__":
    main()
