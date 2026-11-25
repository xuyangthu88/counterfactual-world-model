"""A shared phase decoder for the three contact-action hand-offs in both domains."""

import torch
from torch import nn
from whatif.crossbench.models import mlp
from whatif.shared.model import represent, body_delta, pose
from whatif.factual.model import FactualGraph
from .collect import OUT


class PhaseGraph(FactualGraph):
    def __init__(self):
        super().__init__()
        self.phases = mlp(48, 21, 128)
        nn.init.zeros_(self.phases[-1].weight)
        nn.init.zeros_(self.phases[-1].bias)

    def transition(self, s, a, k, u, keep=None):
        z, edges, peak, rank = super().forward(s, a, k, u, keep)
        coarse = body_delta(s, z)
        x = torch.cat((represent(s, a), coarse, edges.sigmoid(), k[..., None], u[..., None]), -1)
        raw = self.phases(x)
        approach = raw[..., :3]
        stroke = approach + coarse + raw[..., 3:6]
        # Response at the hand-off, in the initial body frame per native observation tick.
        speed = raw[..., 6:9]
        rho = s[..., 17] * s[..., 18] + (1 - s[..., 18]) * (0.48 + 0.39 * k)
        duration = (rho + rho.square() + rho.pow(3)) * s[..., 13] + (1 - s[..., 13])
        end = stroke + speed * duration[..., None] + raw[..., 9:12]
        result = z.clone()
        result[..., :3] = pose(s, end)
        result[..., 14:17] = z[..., 14:17] + raw[..., 12:15]
        result[..., 3:9] = z[..., 3:9] + raw[..., 15:21]
        return result, edges, peak, rank, (approach, stroke, speed)

    def forward(self, s, a, k, u, keep=None):
        return self.transition(s, a, k, u, keep)[:4]


def install(model, device, names=("graph", "pets", "cophy")):
    if "graph" in names:
        net = PhaseGraph().to(device)
        net.load_state_dict(torch.load(OUT / "graph.pt", map_location=device, weights_only=True))
        model.graph = net.eval()
    for key in ("pets", "cophy"):
        if key in names:
            getattr(model, key).load_state_dict(
                torch.load(OUT / f"{key}.pt", map_location=device, weights_only=True)
            )
