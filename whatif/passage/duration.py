"""Native clock of a contact macro, trained on recorded hand-off timestamps."""

import json
import torch
from torch.nn import functional as F
from whatif.crossbench.models import mlp
from whatif.shared.model import represent
from whatif.precision.evaluate import panel
from .model import OUT


def features(s, a):
    rep = represent(s, a)
    return torch.cat((rep[..., 9:22], rep[..., 22:40]), -1)


def pack(t, split):
    d = panel(t, split, "cpu")
    s = d["state"][:, None].expand_as(d["end"])
    return features(s, d["cmd"]).reshape(-1, 31), d["phase_ticks"][..., -1].flatten().float() / 42


def run():
    torch.manual_seed(11)
    torch.set_num_threads(2)
    data = [pack(t, "train") for t in ("pusht", "near", "causalworld")]
    validation = [pack(t, "val") for t in ("pusht", "near", "causalworld")]
    x, y = [torch.cat(xx) for xx in zip(*data)]
    mu, sd = x.mean(0), x.std(0).clamp_min(0.05)
    n = mlp(31, 1, 48)
    opt = torch.optim.AdamW(n.parameters(), lr=0.0015)
    best = 1.0
    stats = []
    for ep in range(45):
        for ix in torch.randperm(len(y)).split(384):
            pred = n((x[ix] - mu) / sd).squeeze(-1)
            loss = F.smooth_l1_loss(pred, y[ix], beta=0.08)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            row = [
                float((n((xx - mu) / sd).squeeze(-1).clamp(3 / 42, 1) - yy).abs().mean())
                for xx, yy in validation
            ]
            stats.append([v * 42 for v in row])
            if sum(row) < best:
                best = sum(row)
                torch.save(dict(weights=n.state_dict(), mu=mu, sd=sd), OUT / "duration.pt")
    (OUT / "duration-fit.json").write_text(json.dumps(stats) + "\n")


def duration(m, s, a):
    if not hasattr(m, "clock_net"):
        d = torch.load(OUT / "duration.pt", map_location=s.device, weights_only=True)
        m.clock_net = mlp(31, 1, 48).to(s.device).eval()
        m.clock_net.load_state_dict(d["weights"])
        m.clock_mu = d["mu"]
        m.clock_sd = d["sd"]
    f = features(s, a)
    return torch.where(
        s[..., 13] > 0.5,
        42 * m.clock_net((f - m.clock_mu) / m.clock_sd).squeeze(-1).clamp(3 / 42, 1),
        42.0,
    )


if __name__ == "__main__":
    run()
