import sys
import json
import torch

from occ.vit import ViTModel
from occ.run_model import run_model
from occ.utils import seed_all, print_elements, hook_attn_map

from functools import partial


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

    # apply a seed for reproducibility
    seed_all(cfg["general"]["seed"])

    if cfg["general"]["verbose"]:
        print("Print config.json items")
        print_elements(cfg)
        print("End config print")

    model = ViTModel(config=cfg)

    attn_maps = []
    if cfg["output"]["error_analysis"] in [
        "attention_overlay",
        "all_attention_overlay",
    ]:
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
