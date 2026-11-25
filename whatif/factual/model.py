"""One posterior over the same five response particles used by the shared contact graph."""

import numpy as np
import torch
from torch import nn
from whatif.crossbench.models import mlp
from whatif.shared.model import SharedGraph, body_delta, physical, represent
from .collect import DATA, OUT


def evidence(previous, action, observed):
    y = body_delta(previous, observed)
    nominal = physical(previous, action, torch.zeros_like(action[..., 0]))
    return torch.cat(
        (
            represent(previous, action),
            y,
            y - nominal,
            observed[..., 14:19],
            torch.ones_like(y[..., :1]),
        ),
        -1,
    )


class FactualGraph(SharedGraph):
    def __init__(self):
        super().__init__()
        self.encoder = mlp(52, 5, 64)

    def belief(self, context):
        return 0.96 * self.encoder(context).softmax(-1) + 0.04 / 5


def read_panel(path, device):
    with np.load(path) as archive:
        d = dict(archive)
    if "fact" not in d:
        d["fact"] = d["previous_actions"][np.arange(len(d["b"])), d["b"]]
    fields = [
        "previous",
        "fact",
        "state",
        "cmd",
        "end",
        "edge",
        "peak",
        "ref",
        "seed",
        "physics",
        "protocol",
    ]
    d = {k: torch.as_tensor(d[k], device=device) for k in fields}
    return {k: (v.float() if v.is_floating_point() else v) for k, v in d.items()}


def cases(task, split, device):
    if task == "wall_crossed" and split == "train":
        task = "wall"
    d = read_panel(DATA / f"{task}-{split}.npz", device)
    if split == "train" and task != "wall":
        past = read_panel(f"data/prospective/{task}-fit.npz", device)
        d = {k: torch.cat((d[k], past[k])) for k in d}
    return d


def selected(d, ix):
    return {k: v[ix] for k, v in d.items()}


class AdaptedModels:
    def __init__(self, device):
        from whatif.shared.planning import Models
        from mbrl.models import GaussianMLP

        self.base = Models(device)
        self.graph = FactualGraph().to(device)
        self.graph.load_state_dict(
            torch.load(OUT / "graph.pt", map_location=device, weights_only=True)
        )
        self.graph.eval()
        self.cophy = self.base.cophy
        self.cophy.load_state_dict(
            torch.load(OUT / "cophy.pt", map_location=device, weights_only=True)
        )
        self.ncii = self.base.ncii
        self.ncii.load_state_dict(
            torch.load(OUT / "ncii.pt", map_location=device, weights_only=True)
        )
        self.pets = GaussianMLP(92, 16, device, num_layers=3, ensemble_size=5, hid_size=144)
        ck = torch.load(OUT / "pets.pt", map_location=device, weights_only=True)
        self.pets.load_state_dict(ck.pop("net"))
        self.pets.eval()
        for key, value in ck.items():
            setattr(self, key, value)
        self.cem = self.base.cem
        self.start = self.base.start
        self.value_readout = self.base.value_readout

    def predict(self, s, a, context=None):
        ctx = self.context if context is None else context
        x = torch.cat((represent(s, a), ctx.expand(*s.shape[:-1], 52)), -1)
        mu, lv = self.pets((x - self.xm) / self.xs, use_propagation=False)
        return mu * self.ys + self.ym, (lv * 0.5).exp() * self.ys
