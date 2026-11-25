"""Fine-tune the existing graph with within-state, action-dependent contact supervision."""

import argparse
import json
import torch
from torch.nn import functional as F
from whatif.shared.model import SharedGraph, panel
from whatif.shared.train import losses
from .collect import OUT


def contact_terms(logits, labels):
    positive = labels.float()
    count = positive.sum(1)
    # Each state/source whose contact can change gets equal weight per outcome.
    switched = (count > 0) & (count < labels.shape[1])
    pos = (F.softplus(-logits) * positive).sum(1) / count.clamp_min(1)
    neg = (F.softplus(logits) * (1 - positive)).sum(1) / (labels.shape[1] - count).clamp_min(1)
    balanced = ((pos + neg) * switched).sum() / (2 * switched.sum().clamp_min(1))
    pairs = positive[:, :, None] * (1 - positive[:, None, :])
    delta = logits[:, :, None] - logits[:, None, :]
    ranking = (F.softplus(-delta) * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)
    return balanced, (ranking * switched).sum() / switched.sum().clamp_min(1)


def objective(net, d, idx):
    s = d["state"][idx]
    a = d["cmd"][idx]
    k = (d["physics"][idx].float() - 1)[:, None].expand_as(a[..., 0])
    u = (d["noise"][idx] / 0.04)[:, None].expand_as(k)
    logits = net(s[:, None].expand(-1, a.shape[1], -1), a, k, u)[1]
    # Equal benchmark weight for switch events irrespective of their prevalence.
    terms = []
    for domain in (0, 1):
        mask = s[:, 13] == domain
        if mask.any():
            ce, rank = contact_terms(logits[mask], d["edge"][idx][mask])
            terms.append(0.32 * ce + 0.18 * rank)
    return losses("graph", net, d, idx, 0) + torch.stack(terms).mean()


def main(device):
    torch.manual_seed(11)
    torch.set_num_threads(2)
    d, v = [panel(x, device) for x in ("train", "val")]
    net = SharedGraph().to(device)
    net.load_state_dict(
        torch.load("results/shared/graph.pt", map_location=device, weights_only=True)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=0.00065)
    best = float("inf")
    history = []
    for epoch in range(48):
        net.train()
        for idx in torch.randperm(len(d["state"]), device=device).split(80):
            loss = objective(net, d, idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 6.0)
            opt.step()
        if epoch in (27, 39):
            for group in opt.param_groups:
                group["lr"] *= 0.38
        if epoch % 3 == 2:
            net.eval()
            with torch.no_grad():
                # Mix both tasks in each validation minibatch before averaging.
                gen = torch.Generator(device=device).manual_seed(11)
                order = torch.randperm(len(v["state"]), generator=gen, device=device)
                val = sum(len(j) * objective(net, v, j).item() for j in order.split(96)) / len(
                    order
                )
            history.append({"epoch": epoch + 1, "objective": val})
            if val < best:
                best = val
                torch.save(net.state_dict(), OUT / "graph.pt")
            print(epoch + 1, round(val, 3), flush=True)
    (OUT / "training.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args().device)
