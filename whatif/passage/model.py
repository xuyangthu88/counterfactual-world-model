"""Body-frame action response and within-stroke goal passage on the shared contact graph."""

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from whatif.crossbench.models import mlp, wrap
from whatif.shared.model import body_delta, pose, represent, physical
from whatif.precision.model import PhaseGraph
from whatif.precision.geometry import overlap
from whatif.precision.model import install as install_precision

OUT = Path("results/passage")


def square_cover(p, s):
    span = 0.5 * (p[..., 2].cos().abs() + p[..., 2].sin().abs())
    rel = p[..., :2] - s[..., 9:11]
    hi = (rel + span[..., None]).clamp(max=0.5)
    lo = (rel - span[..., None]).clamp(min=-0.5)
    return (hi - lo).clamp_min(0).prod(-1)


def native_cover(p, s):
    return torch.where(s[..., 13] > 0.5, overlap(p, s[..., 9:12]), square_cover(p, s))


def closest_coverage(s, handoffs):
    # Find the projection of the goal onto each piece in a translation/angle metric.
    # Query native planar overlap at each projected pose and at all three ends.
    paths = torch.stack(handoffs, -2)
    starts = torch.cat((torch.zeros_like(paths[..., :1, :]), paths[..., :-1, :]), -2)
    delta = paths - starts
    dm = torch.cat((delta[..., :2], wrap(delta[..., 2:3])), -1)
    goal = body_delta(s, s[..., 9:12])[..., None, :] - starts
    goal = torch.cat((goal[..., :2], wrap(goal[..., 2:3])), -1)
    scale = s.new_tensor([1.0, 1.0, 0.47])
    frac = (
        (goal * scale * dm * scale).sum(-1) / (dm * scale).square().sum(-1).clamp_min(1e-8)
    ).clamp(0, 1)
    d = torch.cat((starts + frac[..., None] * dm, paths), -2)
    expanded = s.unsqueeze(-2).expand(*d.shape[:-1], 19)
    scores = native_cover(pose(expanded, d), expanded)
    return scores[..., -1], torch.maximum(s[..., 12], scores.amax(-1))


def physical_history(ctx):
    # The factual stroke's observed motion, residual from nominal, and observed speed.
    return torch.cat((ctx[..., 40:46], ctx[..., 46:51]), -1)


class PassageGraph(PhaseGraph):
    def __init__(self):
        super().__init__()
        # Stage 12 already predicts the three hand-offs. Correct their body-frame motion
        # with a shared local response field conditioned on the completed factual stroke.
        self.response = mlp(57, 9, 112)
        self.passage = mlp(17, 2, 64)
        self.register_buffer("location", torch.zeros(57))
        self.register_buffer("scale", torch.ones(57))
        nn.init.zeros_(self.response[-1].weight)
        nn.init.zeros_(self.response[-1].bias)
        nn.init.zeros_(self.passage[-1].weight)
        nn.init.zeros_(self.passage[-1].bias)
        self.context = None

    def extract(self, s, a, k, u, keep=None):
        z, e, pk, rank, (da, db, speed) = super().transition(s, a, k, u, keep)
        end = body_delta(s, z)
        rep = represent(s, a)
        # Components in the object's frame: domain, velocity, action and contact sources.
        local = torch.cat(
            (
                rep[..., 9:22],
                rep[..., 22:40],
                da,
                db,
                end,
                e.sigmoid(),
                k[..., None],
                u[..., None],
                physical(s, a, k)[..., :1],
            ),
            -1,
        )
        return z, e, pk, rank, local

    def decode(self, local, history):
        x = torch.cat((local, history.expand(*local.shape[:-1], 11)), -1)
        correction = self.response(((x - self.location) / self.scale).clamp(-12, 12))
        hand = local[..., 31:40] + correction
        return hand[..., :3], hand[..., 3:6], hand[..., 6:9]

    def read_passage(self, s, local, history, hands):
        # Geometry provides task units and an initial location. Native stepwise maxima
        # supervise its residual and predictive spread, including transits between hand-offs.
        end, candidate = closest_coverage(s, hands)
        h = history.expand(*local.shape[:-1], 11)
        feats = torch.cat(
            (h, local[..., 40:43], s[..., 13:14], s[..., 12:13], candidate[..., None]), -1
        )
        raw = self.passage(feats)
        loc = (candidate + 0.16 * raw[..., 0].tanh()).clamp(0, 1)
        loc = torch.maximum(loc, s[..., 12])
        width = 0.004 + 0.032 * F.softplus(raw[..., 1])
        threshold = 0.9 + 0.05 * s[..., 13]
        prob = ((loc - threshold) / width).sigmoid()
        return end, loc, width, prob

    def predict_event(self, s, a, k, u, history=None, keep=None):
        z, e, pk, rank, local = self.extract(s, a, k, u, keep)
        if history is None:
            history = self.context
        if history is None:
            history = s.new_zeros(11)
        hands = self.decode(local, history)
        end, peak, width, prob = self.read_passage(s, local, history, hands)
        z = z.clone()
        z[..., :3] = pose(s, hands[-1])
        z[..., 12] = torch.where(s[..., 13] > 0.5, end, z[..., 12])
        return z, e, peak, rank, prob

    def forward(self, s, a, k, u, keep=None):
        return self.predict_event(s, a, k, u, keep=keep)[:4]


def passage_grid(graph, s, a, context):
    pair = torch.cartesian_prod(
        s.new_tensor([-1.0, -0.5, 0.0, 0.5, 1.0]), s.new_tensor([-1.0, 0.0, 1.0])
    )
    shape = (*a.shape[:-1], len(pair))
    st = s[:, None, None].expand(*shape, 19)
    cmd = a[:, :, None].expand(*shape, 7)
    return graph.predict_event(
        st,
        cmd,
        pair[:, 0].expand(shape),
        pair[:, 1].expand(shape),
        physical_history(context)[:, None, None],
    )


def install(m, device):
    install_precision(m, device)
    graph = PassageGraph().to(device)
    graph.load_state_dict(torch.load(OUT / "graph.pt", weights_only=True, map_location=device))
    m.graph = graph.eval()
