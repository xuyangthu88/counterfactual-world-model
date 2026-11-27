"""Stage 33: Evaluate shared 5-domain world models and original baselines across all 5 benchmarks."""
import argparse
import json

import torch

from whatif.effects.baseline import cophy_predict, pets_predict
from whatif.effects.data import subset
from whatif.effects.evaluate import combine, contact_graph, edits, prediction
from whatif.effects.ncii import predict as ncii_predict
from whatif.shortstep.evaluate import native_metrics, pushing_edits
from .baselines import restore_cophy, restore_ncii, restore_pets
from .data import ALL_DOMAINS, NEW_DOMAINS, UNIT, load, native_path
from .fit import CONFIGS, OUT, build_model

KEEP_KEYS = ('pose', 'edge', 'causal', 'reward', 'w', 'p_dist', 'r_dist')

# Patch UNIT in whatif.effects.evaluate and whatif.shortstep.evaluate so scoring handles all 5 domains
import whatif.effects.evaluate as _eff_eval
import whatif.shortstep.evaluate as _ss_eval
_eff_eval.UNIT.update(UNIT)
_ss_eval.UNIT.update(UNIT)


def restore_world(name, device):
    kind = name.removesuffix('-calibrated')
    model = build_model(kind).to(device)
    model.load_state_dict(torch.load(OUT / f'{name}.pt', map_location=device)['model'])
    return model.eval()


def transfer_split_name(domain):
    return 'transfer' if domain in NEW_DOMAINS else 'cumulativetransfer'


@torch.no_grad()
def evaluate_all(models, domains, device='cuda'):
    torch.set_num_threads(2)
    results = {}
    for domain in domains:
        tr_split = transfer_split_name(domain)
        splits = ('test', tr_split)
        sets = {s: load(domain, s, device, labels=False) for s in ('val', *splits)}
        results[domain] = {s: {} for s in splits}

        for name in models:
            if name == 'cophynet':
                model = restore_cophy(domain, device)
                flavor = 'cophynet'
                run = lambda d, m=model: {'pose': cophy_predict(m, d)['pose'], 'reward': None}
            elif name == 'pets':
                model = restore_pets(domain, device)
                flavor = 'pets'
                torch.manual_seed(345)
                rng = torch.Generator(device=device).manual_seed(345)
                run = lambda d, m=model, r=rng: pets_predict(m, d, r)
            elif name == 'ncii':
                model = restore_ncii(domain, device)
                flavor = 'ncii'
                run = lambda d, m=model: ncii_predict(m, d)
            else:
                model = restore_world(name, device)
                flavor = 'ours'
                run = lambda d, m=model: {k: v for k, v in m(d).items() if k in KEEP_KEYS}

            def infer(d):
                return combine([run(subset(d, ix)) for ix in torch.arange(len(d['x']), device=device).split(16)])

            cuts = {}
            if flavor in ('ours', 'ncii'):
                val_out = infer(sets['val'])
                val_g = contact_graph(sets['val'], val_out, {})
                cuts = {k: v['cut'] for k, v in val_g.items() if isinstance(v, dict) and 'cut' in v}

            for split in splits:
                d = sets[split]
                o = infer(d)
                row = {}
                if flavor != 'ncii':
                    row.update(prediction(d, o, domain, flavor))
                if flavor in ('ours', 'ncii'):
                    row['graph'] = contact_graph(d, o, cuts)
                if flavor == 'ours':
                    if 'off' in d:
                        row['edit'] = edits(model, d)
                    elif native_path(domain, split).exists():
                        d_full = load(domain, split, device, labels=True)
                        row['edit'] = pushing_edits(model, d_full, domain)
                    if native_path(domain, split).exists():
                        d_full = load(domain, split, device, labels=True)
                        row['native'] = native_metrics(model, d_full, domain)
                results[domain][split][name] = row
                delta = row.get('delta_rmse')
                ap = row.get('graph', {}).get('contact', {}).get('ap')
                print(f'{domain:11s} {split:18s} {name:26s} delta={delta} ap={ap}', flush=True)

    # Summary vs better of original CoPhyNet / PETS
    summary = {}
    for domain in domains:
        summary[domain] = {}
        for split in ('test', transfer_split_name(domain)):
            tbl = results[domain][split]
            if 'cophynet' in tbl and 'pets' in tbl:
                cp = tbl['cophynet']['delta_rmse']
                pt = tbl['pets']['delta_rmse']
                best_base = min(cp, pt)
                best_name = 'cophynet' if cp <= pt else 'pets'
            else:
                best_base, best_name = None, None
            row_sum = {'best_baseline': best_name, 'best_baseline_delta_rmse': best_base}
            for name, m in tbl.items():
                if 'delta_rmse' in m and best_base:
                    red = 100.0 * (best_base - m['delta_rmse']) / best_base
                    row_sum[name] = dict(
                        delta_rmse=round(m['delta_rmse'], 4),
                        reduction_pct=round(red, 2),
                        contact_ap=round(m['graph']['contact']['ap'], 4) if 'graph' in m and m['graph']['contact']['ap'] is not None else None,
                        contact_ber=round(m['graph']['contact']['ber'], 4) if 'graph' in m and m['graph']['contact']['ber'] is not None else None,
                    )
            summary[domain][split] = row_sum

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return results, summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--domains', nargs='+', default=list(ALL_DOMAINS))
    p.add_argument('--models', nargs='+')
    args = p.parse_args()
    avail = [f'{k}-calibrated' for k in CONFIGS if (OUT / f'{k}-calibrated.pt').exists()]
    models = args.models or [*avail, 'cophynet', 'pets', 'ncii']
    evaluate_all(models, args.domains, args.device)
