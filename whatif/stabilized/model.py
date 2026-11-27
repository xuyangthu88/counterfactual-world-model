"""Stabilized counterfactual world model with temporal kinematic abduction,
decoupled historical prior, SE(3) local-frame geometry, inertial-impulse
velocity integration, and shared within-step multi-hop constraint propagation."""
import torch
from torch import nn
from torch.nn import functional as F

from whatif.effects.data import clock, pair_ids
from whatif.effects.model import COMPONENTS, FACTORS, average, relative
from whatif.mechanism.model import K, U, mlp, pairmask, pairs, pool, tile


def sparse_gate(p, threshold=0.08, soft=0.0):
    """Exact zero below threshold, continuous confidence above, with smooth STE gradient."""
    active = (p >= threshold).to(p)
    ste = p * active + 0.5 * (p - p.detach()) * (1.0 - active)
    return soft * p + (1.0 - soft) * ste


def safe_pose_input(pose, bound_z=False):
    """Optionally replace raw vertical height z with ground-proximity encoding to avoid height OOD."""
    if not bound_z:
        return pose
    z = pose[..., 2:3]
    z_enc = 1.0 - torch.exp(-z.clamp_min(0.0)) + z.clamp_max(0.0)
    return torch.cat([pose[..., :2], z_enc, pose[..., 3:]], dim=-1)


def pose_frame(pose):
    """Construct an orthonormal 3x3 frame [e1, e2, e3] per object from 9D pose [xyz, r1, r2]."""
    r1 = pose[..., 3:6]
    r2 = pose[..., 6:9]
    n1 = r1.norm(dim=-1, keepdim=True)
    default1 = torch.zeros_like(r1)
    default1[..., 0] = 1.0
    e1 = torch.where(n1 > 1e-4, r1 / n1.clamp_min(1e-4), default1)

    u2 = r2 - (r2 * e1).sum(dim=-1, keepdim=True) * e1
    n2 = u2.norm(dim=-1, keepdim=True)
    # Fallback axis orthogonal to e1 if r2 is degenerate (e.g., padded absent node)
    alt = torch.zeros_like(r2)
    alt[..., 1] = 1.0
    alt_fallback = torch.zeros_like(r2)
    alt_fallback[..., 2] = 1.0
    alt = torch.where(e1[..., 1:2].abs() < 0.9, alt, alt_fallback)
    alt = alt - (alt * e1).sum(dim=-1, keepdim=True) * e1
    alt = alt / alt.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    e2 = torch.where(n2 > 1e-4, u2 / n2.clamp_min(1e-4), alt)
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    return e1, e2, e3


def local_pair_geometry(pose_prop, vel):
    """Rotation-invariant 24D rigid-body pairwise geometry for all ordered pairs (i -> j)."""
    e1, e2, e3 = pose_frame(pose_prop)
    # Stack frame basis: (..., N, 3_axes, 3_xyz)
    R = torch.stack([e1, e2, e3], dim=-2)
    dp = relative(pose_prop[..., :3])  # (..., i, j, 3) = p_j - p_i
    dv = relative(vel[..., :3])        # (..., i, j, 3) = v_j - v_i
    # Project dp and dv into sender i frame and receiver j frame
    q_i = torch.einsum('...iad,...ijd->...ija', R, dp)
    q_j = torch.einsum('...jad,...ijd->...ija', R, dp)
    u_i = torch.einsum('...iad,...ijd->...ija', R, dv)
    u_j = torch.einsum('...jad,...ijd->...ija', R, dv)
    rel_R = torch.einsum('...iad,...jbd->...ijab', R, R).flatten(-2)  # 9D
    dist = dp.norm(dim=-1, keepdim=True)
    rel_z = dp[..., 2:3]
    rel_spd = dv.norm(dim=-1, keepdim=True)
    return torch.cat([q_i, q_j, u_i, u_j, rel_R, dist, rel_z, rel_spd], dim=-1), (e1, e2, e3)


class TemporalFacts(nn.Module):
    """Infer physical mechanism weights m, Gaussian disturbance u, and revised factual graph
    from multi-step kinematic trajectories (positions, velocities, accelerations, speed ratios)."""

    def __init__(self, feature=96, bound_z=False):
        super().__init__()
        self.feature = feature
        self.bound_z = bound_z
        # Inputs per step: pose(9) + cmd(7) + role(3) + clock(2) + vel(9) + acc(9) + spd(1) + ratio(1) = 41
        self.local = mlp(41, feature, feature)
        self.prior = mlp(2 * feature + 4, 1, feature)
        self.motion = mlp(2 * feature + 9, 9, feature)
        self.step_enc = mlp(feature + 27, feature, feature)
        self.temporal = nn.GRU(feature, feature, batch_first=True)
        self.history_norm = nn.LayerNorm(feature)
        self.attend = nn.Linear(feature, 1)
        self.latent = mlp(feature + 1, K + 2 * U, feature)
        self.mechanisms = nn.Parameter(torch.randn(K, 12) * 0.25)
        self.correct = mlp(2 * (feature + 12 + U + 9) + 1, 1, feature)
        self.refine = mlp(feature + 2, K + 2 * U, feature)

    def forward(self, d):
        role, factual = d['role'], d['fact']
        current, next_pose = factual[:, :-1], factual[:, 1:]
        b, t, n, _ = current.shape
        dt = d['widths'][:, :, None, None].clamp_min(1e-3)
        vel = (next_pose - current) / dt
        prev_vel = torch.cat([torch.zeros_like(vel[:, :1]), vel[:, :-1]], dim=1)
        acc = vel - prev_vel
        speed = vel[..., :3].norm(dim=-1, keepdim=True)
        prev_speed = prev_vel[..., :3].norm(dim=-1, keepdim=True)
        ratio = (speed / (prev_speed + 0.05)).clamp(0.0, 4.0)

        command = d['fa'][:, None].expand(-1, t, -1, -1)
        pose_in = safe_pose_input(current, self.bound_z)
        role_t = role[:, None].expand(-1, t, -1, -1)
        clk_t = clock(d['widths'])[:, :, None].expand(-1, -1, n, -1)
        local = self.local(torch.cat([pose_in, command, role_t, clk_t, vel, acc, speed, ratio], dim=-1))

        rel_p = relative(current[..., :3])
        rel_d = rel_p.norm(dim=-1, keepdim=True)
        prior = self.prior(torch.cat([pairs(local), rel_p, rel_d], dim=-1))[..., 0]
        valid = pairmask(role)
        prob = prior.sigmoid() * valid[:, None]
        message = torch.einsum('btij,btih->btjh', prob, local)
        motion = self.motion(torch.cat([local, message / prob.sum(-2).clamp_min(1.0)[..., None], vel], dim=-1))
        innovation = next_pose - current - motion

        enc = self.step_enc(torch.cat([local, vel, motion, innovation], dim=-1))
        # Temporal GRU per node preserves deceleration / cascade order
        seq = enc.transpose(1, 2).reshape(b * n, t, self.feature)
        _, h_last = self.temporal(seq)
        history = self.history_norm(h_last.squeeze(0).reshape(b, n, self.feature))

        alpha_t = self.attend(enc).softmax(dim=1)
        innovation_avg = (innovation * alpha_t).sum(dim=1)
        alpha_scalar = alpha_t[..., 0]
        initial = (prior * (alpha_scalar[..., None] + alpha_scalar[:, :, None, :]) * 0.5).sum(dim=1)
        initial = 0.5 * (initial + initial.transpose(-1, -2))

        def degree(logit):
            return (logit.sigmoid() * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1.0)

        def posterior(v):
            coef, mean, log_sd = v.split([K, U, U], dim=-1)
            probs = coef.softmax(-1)
            return probs @ self.mechanisms, mean, log_sd.clamp(-4.0, 1.0), probs

        first = self.latent(torch.cat([history, degree(initial)], dim=-1))
        m0, u0, _, _ = posterior(first)
        edit = torch.cat([pairs(torch.cat([history, m0, u0, innovation_avg], dim=-1)), initial[..., None]], dim=-1)
        factual_graph = initial + self.correct(edit)[..., 0]
        factual_graph = 0.5 * (factual_graph + factual_graph.transpose(-1, -2))

        refined = first + self.refine(torch.cat([history, degree(initial), degree(factual_graph)], dim=-1))
        m, mean, scale, probs = posterior(refined)
        eps = d.get('eps')
        if eps is None:
            eps = torch.randn_like(mean) if self.training else torch.zeros_like(mean)
        kl = 0.5 * (mean.square() + (2.0 * scale).exp() - 1.0 - 2.0 * scale)

        # Batch-level mechanism diversity entropy across present movable nodes
        mover_mask = role[..., 0] > 0.5
        if mover_mask.any():
            mean_probs = probs[mover_mask].mean(dim=0)
            mech_entropy = -(mean_probs * mean_probs.clamp_min(1e-9).log()).sum()
        else:
            mech_entropy = probs.new_tensor(0.0)

        return dict(
            m=m,
            u=mean + eps * scale.exp(),
            graph=factual_graph,
            kl=average(kl, role.sum(-1)[..., None]),
            innovation=innovation,
            mech_entropy=mech_entropy,
        )


class StabilizedWorld(nn.Module):
    """Shared counterfactual world model supporting:
      - mode='fix_only': implementation fixes only (1-hop, world relative coords, pose-step)
      - mode='inertial_multihop': full inertial-impulse velocity dynamics + SE(3) local-frame
        geometry + shared within-step multi-hop constraint iterations (hops=2 or 3).
    """

    def __init__(self, mode='inertial_multihop', hops=2, width=32, feature=96, gate_threshold=0.08):
        super().__init__()
        assert mode in ('fix_only', 'inertial_multihop')
        self.mode = mode
        self.hops = 1 if mode == 'fix_only' else int(hops)
        self.width = width
        self.feature = feature
        self.gate_threshold = gate_threshold
        self.bound_z = (mode == 'inertial_multihop')

        self.facts = TemporalFacts(feature=feature, bound_z=self.bound_z)
        # node context: pose(9) + fact_disp(9) + role(3) + m(12) + u(4) = 37
        self.node = mlp(9 + 9 + 3 + 12 + U, feature, feature)
        self.factor = nn.ModuleList([mlp(feature + 3, width, feature) for _ in range(FACTORS)])
        # local action injection: cmd(7) + cmd_err(3) + time(2) = 12
        self.action = mlp(7 + 3 + 2, FACTORS * width, feature)

        # Endpoint encoder includes persistent (m, u)
        # state(3*W) + msg(3*W) + pose(9) + vel(9) + free(9) + m(12) + u(4)
        self.endpoint = mlp(2 * FACTORS * width + 27 + 12 + U, feature, feature)

        # Edge interaction network: decoupled prior_feat(1) + geometry
        geom_dim = 24 if mode == 'inertial_multihop' else 4
        edge_in = 2 * feature + geom_dim + 1 + 2 + 2 * (12 + U)
        self.edge = mlp(edge_in, 4 + 2 * FACTORS**2, max(feature, 112))
        self.message = nn.Linear(width, width, bias=False)

        # Bounded GRUCell + LayerNorm conditioned on (message, m, u, time)
        self.recurrent = nn.GRUCell(width + 12 + U + 2, width)
        self.state_norm = nn.LayerNorm(width)

        # Motion proposal and graph-gated response
        # propose input per factor: state(W) + pose_f(3) + vel_f(3) + cmd_err(3) + m(12) + u(4) + time(2)
        self.propose = nn.ModuleList([mlp(width + 9 + 12 + U + 2, 3, feature) for _ in range(FACTORS)])
        if mode == 'inertial_multihop':
            # Per-factor velocity retention logit conditioned on (state, m, u, time)
            self.retention = nn.ModuleList([mlp(width + 12 + U + 2, 1, feature // 2) for _ in range(FACTORS)])
            for head in self.retention:
                nn.init.zeros_(head[-1].weight)
                nn.init.constant_(head[-1].bias, 0.5)
        else:
            self.retention = None

        # Response head conditioned on (h, msg, m, u, time), zeroed at msg=0 by difference
        self.response = nn.ModuleList([mlp(2 * width + 12 + U + 2, 3, feature) for _ in range(FACTORS)])
        for obj in (*self.propose, *self.response):
            nn.init.normal_(obj[-1].weight, std=0.02)
            nn.init.zeros_(obj[-1].bias)

        # Spatial vector message on position-factor weights
        self.n_dirs = 10 if mode == 'inertial_multihop' else 6
        vec_in = 2 * feature + geom_dim + 2 * (12 + U) + 2
        self.vector = mlp(vec_in, FACTORS * self.n_dirs, max(feature, 96))
        nn.init.zeros_(self.vector[-1].weight)
        nn.init.zeros_(self.vector[-1].bias)

        # Reward & coupled 4-component Gaussian mixture heads
        self.reward = mlp(FACTORS * width + 18, feature, feature)
        self.reward_mean = nn.Linear(feature, 1)
        self.regime = mlp(feature, COMPONENTS, feature)
        self.position_dist = mlp(width, 3 * 3 * COMPONENTS, feature)
        self.reward_dist = nn.Linear(feature, 3 * COMPONENTS)

    def _pair_geom(self, pose_prop, vel):
        if self.mode == 'inertial_multihop':
            return local_pair_geometry(pose_prop, vel)
        dp = relative(pose_prop[..., :3])
        dist = dp.norm(dim=-1, keepdim=True)
        return torch.cat([dp, dist], dim=-1), None

    def interaction(self, end, pose_prop, vel, prior_feat, mu_pair, dt_clk, mask, omit, soft):
        n = pose_prop.size(1)
        geom, frames = self._pair_geom(pose_prop, vel)
        features = torch.cat([
            pairs(end), geom, prior_feat[..., None],
            dt_clk[:, None, None].expand(-1, n, n, -1), mu_pair,
        ], dim=-1)
        p, c, strength, persist, opening, value = self.edge(features).split([1, 1, 1, 1, 9, 9], dim=-1)
        # Decoupled: no bare addition of unbounded prior!
        logit = 0.5 * (p[..., 0] + p[..., 0].transpose(-1, -2))
        contact = logit.sigmoid()
        conduct = (c[..., 0] + 1.5).sigmoid()
        magnitude = F.softplus(strength[..., 0])
        persist_p = persist[..., 0].sigmoid()

        gate = sparse_gate(contact, self.gate_threshold, soft)
        edge_gate = mask * omit * gate * conduct * (1.0 + magnitude.tanh())
        factor_gate = opening.sigmoid() * value.tanh()
        weights = edge_gate[..., None, None] * factor_gate.unflatten(-1, (FACTORS, FACTORS))
        return weights, logit, conduct, persist_p, magnitude, geom, frames

    def vector_message(self, end, pose, proposed, vel, free_pos, geom, frames, mu_pair, time, weights):
        n = pose.size(1)
        x = pose[..., :3]
        q = proposed[..., :3]
        v = vel[..., :3]

        def norm_v(z):
            return z / (1.0 + z.norm(dim=-1, keepdim=True))

        def endpoints(z):
            return z[:, :, None].expand(-1, -1, n, -1), z[:, None].expand(-1, n, -1, -1)

        if self.mode == 'inertial_multihop':
            e1, e2, e3 = frames
            e1_i, e1_j = endpoints(e1)
            e2_i, e2_j = endpoints(e2)
            _, e3_j = endpoints(e3)
            v_i, v_j = endpoints(v)
            directions = torch.stack([
                norm_v(relative(x)), norm_v(relative(q)),
                norm_v(v_i), norm_v(v_j), norm_v(relative(v)),
                e1_i, e2_i, e1_j, e2_j, e3_j,
            ], dim=-2)
        else:
            directions = torch.stack([
                norm_v(relative(x)), norm_v(relative(q)),
                *endpoints(norm_v(v)), *endpoints(norm_v(free_pos)),
            ], dim=-2)

        pair_in = torch.cat([pairs(end), geom, mu_pair, time[:, None].expand(-1, n, -1, -1)], dim=-1)
        coef = self.vector(pair_in).unflatten(-1, (FACTORS, self.n_dirs))
        return torch.einsum('bijf,bijfl,bijlc->bjc', weights[..., 0], coef, directions)

    def rollout(self, d, node, facts, action, reset, keep=None, soft=0.0):
        x, roles = d['x'], d['role']
        b, k, n = action.shape[:3]
        x0 = tile(x, k)
        reset_flat = reset.flatten(0, 1)
        pose = x0 + reset_flat
        body = tile(node, k)
        cmd = action.flatten(0, 1)
        mask = tile(pairmask(roles), k)
        omit = torch.ones_like(mask) if keep is None else keep.flatten(0, 1)

        # Decoupled, bounded historical graph feature
        prior_feat = torch.tanh(tile(facts['graph'], k) / 3.0)
        m_tile = tile(facts['m'], k)  # (BK, N, 12)
        u_tile = tile(facts['u'], k)  # (BK, N, 4)
        mu_node = torch.cat([m_tile, u_tile], dim=-1)  # (BK, N, 16)
        mu_pair = pairs(mu_node)                       # (BK, N, N, 32)
        mu_factor = mu_node[:, :, None].expand(-1, -1, FACTORS, -1)

        # Distinguish ongoing waypoint commands from initial-state pose interventions (reset != 0)
        ongoing = cmd[..., 0] * (reset_flat[..., :3].norm(dim=-1) < 1e-6).to(cmd)
        w1 = x0[..., :3] + cmd[..., 1:4]
        w2 = x0[..., :3] + cmd[..., 4:7]

        pose_init_in = safe_pose_input(pose, self.bound_z)
        state = torch.stack([f(torch.cat([body, q], dim=-1))
                             for f, q in zip(self.factor, pose_init_in.split(3, dim=-1))], dim=-2)
        state = self.state_norm(state)
        message = torch.zeros_like(state)

        observed = (d['fact'][:, -1] - d['fact'][:, -2]) * (1.0 - d['same'][:, None, None])
        velocity = tile(observed / d['widths'][:, -1, None, None].clamp_min(1e-3), k)

        widths = d['native_widths']
        nominal = d['nominal_widths']
        elapsed = widths.cumsum(1) / widths.sum(1, keepdim=True)
        times = torch.stack([torch.log1p(nominal / 8.0), elapsed], dim=-1)
        carry = torch.zeros_like(prior_feat)

        outputs = []
        for t_clk, dt_raw, parent_raw in zip(times.unbind(1), widths.unbind(1), nominal.unbind(1)):
            time = tile(t_clk, k)[:, None].expand(-1, n, -1)
            dt = tile(dt_raw, k)[:, None, None]
            parent = tile(parent_raw, k)[:, None, None]
            fraction = (dt / parent).clamp(0.0, 1.0)
            s_prog = time[..., 1:2]  # elapsed fraction in (0, 1]

            # Local piecewise-linear command tracking error ONLY for active ongoing commanded node
            s1 = (2.0 * s_prog).clamp(0.0, 1.0)
            s2 = (2.0 * s_prog - 1.0).clamp(0.0, 1.0)
            target = x0[..., :3] + s1 * (w1 - x0[..., :3]) + s2 * (w2 - w1)
            cmd_err = ongoing[..., None] * (target - pose[..., :3])

            source = self.action(torch.cat([cmd, cmd_err, time], dim=-1)).unflatten(-1, (FACTORS, self.width))
            state = self.state_norm(state + cmd[..., 0, None, None] * source * fraction[..., None])

            pose_in = safe_pose_input(pose, self.bound_z)
            node_time = time[:, :, None].expand(-1, -1, FACTORS, -1)

            if self.mode == 'fix_only':
                last = velocity * parent
                free = torch.stack([
                    f(torch.cat([state[:, :, i], p, dp, cmd_err, mu_node, time], dim=-1))
                    for i, (f, p, dp) in enumerate(zip(self.propose, pose_in.split(3, -1), last.split(3, -1)))
                ], dim=-2)
                proposed = pose + fraction * free.flatten(-2)
                end = self.endpoint(torch.cat([
                    state.flatten(-2), message.flatten(-2), pose_in, last, free.flatten(-2), mu_node,
                ], dim=-1))
                weights, logit, conduct, persist_p, strength, geom, frames = self.interaction(
                    end, proposed, last, prior_feat, mu_pair, time[:, 0], mask, omit, soft)
                contact = logit.sigmoid()
                carry = 1.0 - (1.0 - contact * conduct) * (1.0 - carry * persist_p)
                message = torch.einsum('bijfg,bifh->bjgh', weights, self.message(state))
                physical = self.vector_message(end, pose, proposed, last, free[:, :, 0], geom, frames, mu_pair, time, weights)

                gru_in = torch.cat([message, mu_factor, node_time], dim=-1).flatten(0, 2)
                next_h = self.recurrent(gru_in, state.flatten(0, 2)).reshape_as(state)
                state = self.state_norm((1.0 - fraction[..., None]) * state + fraction[..., None] * next_h)

                change = []
                for i, f in enumerate(self.response):
                    h, m_f = state[:, :, i], message[:, :, i]
                    base_ctx = torch.cat([h, torch.zeros_like(m_f), mu_node, time], dim=-1)
                    full_ctx = torch.cat([h, m_f, mu_node, time], dim=-1)
                    change.append(f(full_ctx) - f(base_ctx))
                future = proposed + fraction * (torch.cat(change, dim=-1) + F.pad(physical, (0, 6)))
                velocity = ((future - pose) / dt.clamp_min(1e-3)).clamp(-50.0, 50.0)
                pose = future
            else:
                # Inertial-impulse velocity dynamics + shared within-step multi-hop constraint solver
                v_free_parts = []
                free_acc_parts = []
                for i, (f_acc, f_ret, p, vf) in enumerate(zip(
                        self.propose, self.retention, pose_in.split(3, -1), velocity.split(3, -1))):
                    ret_base = f_ret(torch.cat([state[:, :, i], mu_node, time], dim=-1)).sigmoid()
                    # Exponential velocity retention across dt steps
                    retain = ret_base.clamp(1e-4, 1.0 - 1e-4).pow(dt)
                    acc = f_acc(torch.cat([state[:, :, i], p, vf, cmd_err, mu_node, time], dim=-1))
                    v_free_parts.append(retain * vf + dt * acc)
                    free_acc_parts.append(acc)
                v_curr = torch.cat(v_free_parts, dim=-1).clamp(-50.0, 50.0)
                free_acc = torch.cat(free_acc_parts, dim=-1)
                proposed = pose + dt * v_curr

                hop_logits, hop_conducts, hop_persists, hop_strengths = [], [], [], []
                alpha_hop = (fraction / self.hops)[..., None]
                for _ in range(self.hops):
                    prop_in = safe_pose_input(proposed, self.bound_z)
                    end = self.endpoint(torch.cat([
                        state.flatten(-2), message.flatten(-2), prop_in, v_curr, free_acc, mu_node,
                    ], dim=-1))
                    weights, logit_h, conduct_h, persist_h, strength_h, geom, frames = self.interaction(
                        end, proposed, v_curr, prior_feat, mu_pair, time[:, 0], mask, omit, soft)
                    message = torch.einsum('bijfg,bifh->bjgh', weights, self.message(state))
                    dv_vec = self.vector_message(
                        end, pose, proposed, v_curr, free_acc[..., :3], geom, frames, mu_pair, time, weights)

                    gru_in = torch.cat([message, mu_factor, node_time], dim=-1).flatten(0, 2)
                    next_h = self.recurrent(gru_in, state.flatten(0, 2)).reshape_as(state)
                    state = self.state_norm((1.0 - alpha_hop) * state + alpha_hop * next_h)

                    dv_parts = []
                    for i, f in enumerate(self.response):
                        h, m_f = state[:, :, i], message[:, :, i]
                        base_ctx = torch.cat([h, torch.zeros_like(m_f), mu_node, time], dim=-1)
                        full_ctx = torch.cat([h, m_f, mu_node, time], dim=-1)
                        dv_parts.append((f(full_ctx) - f(base_ctx)) / self.hops)
                    dv_imp = torch.cat(dv_parts, dim=-1) + F.pad(dv_vec / self.hops, (0, 6))
                    v_curr = (v_curr + dv_imp).clamp(-50.0, 50.0)
                    proposed = pose + dt * v_curr

                    hop_logits.append(logit_h)
                    hop_conducts.append(conduct_h)
                    hop_persists.append(persist_h)
                    hop_strengths.append(strength_h)

                logit = torch.stack(hop_logits, dim=0).amax(dim=0)
                conduct = torch.stack(hop_conducts, dim=0).amax(dim=0)
                persist_p = torch.stack(hop_persists, dim=0).mean(dim=0)
                strength = torch.stack(hop_strengths, dim=0).amax(dim=0)
                contact = logit.sigmoid()
                carry = 1.0 - (1.0 - contact * conduct) * (1.0 - carry * persist_p)

                velocity = v_curr
                pose = proposed

            outputs.append((pose, logit, torch.logit(carry.clamp(1e-5, 1.0 - 1e-5)), strength, state))
        return [torch.stack(v, dim=1).unflatten(0, (b, k)) for v in zip(*outputs)]

    def forward(self, d, removed=None, *, soft=0.0, native=False, detach_dist=True):
        # Center horizontal coordinates on moving objects in shared initial state
        xyz = pool(d['x'][..., :3], d['role'][..., 0])
        shift = F.pad(xyz[..., :2], (0, 7))
        cd = dict(d, x=d['x'] - shift[:, None], fact=d['fact'] - shift[:, None, None], goal=d['goal'] - shift)

        facts = self.facts(cd)
        x, role, action = cd['x'], cd['role'], cd['a']
        node = self.node(torch.cat([
            safe_pose_input(x, self.bound_z), cd['fact'][:, -1] - cd['fact'][:, 0], role, facts['m'], facts['u'],
        ], dim=-1))

        ref = cd['fa'][:, None]
        all_action = torch.cat([action, ref], dim=1)
        all_reset = torch.cat([cd['reset'], torch.zeros_like(cd['reset'][:, :1])], dim=1)
        keep = None
        b, k, n = action.shape[:3]
        if removed is not None:
            variants = removed.size(2)
            pi, pj = pair_ids(n, x.device)
            delete = x.new_zeros(b, k, variants, n, n)
            ix = pi[removed][..., None, None].expand(-1, -1, -1, 1, n)
            delete.scatter_(-2, ix, F.one_hot(pj[removed], n)[..., None, :].to(x))
            delete = delete + delete.transpose(-1, -2)
            keep = torch.cat([x.new_ones(b, k + 1, n, n), 1.0 - delete.flatten(1, 2)], dim=1)
            all_action = torch.cat([all_action, action.repeat_interleave(variants, dim=1)], dim=1)
            all_reset = torch.cat([all_reset, cd['reset'].repeat_interleave(variants, dim=1)], dim=1)

        position, edges, causal, strength, state = self.rollout(cd, node, facts, all_action, all_reset, keep, soft)
        # Unanchored direct prediction avoids adding chaotic factual fall variance to CoPhy pos_RMSE;
        # ref_pose is exposed for optional factual-contrast training loss.
        regular = position[:, :k]
        ref_pose = position[:, k:k + 1]
        state_k = state[:, :k]

        movable = role[:, None, None, :, 0]
        goal = cd['goal'][:, None, None].expand(-1, k, regular.size(2), -1)
        reward_features = self.reward(torch.cat([
            pool(state_k.flatten(-2), movable), pool(regular, movable) - goal, goal,
        ], dim=-1))

        node_for_dist = node.detach() if detach_dist else node
        state_for_dist = state_k.detach() if detach_dist else state_k
        rf_for_dist = reward_features.detach() if detach_dist else reward_features

        w = self.regime(pool(node_for_dist, role.sum(-1))).softmax(-1)

        def distribution(raw):
            offset, scale, load = raw.unflatten(-1, (COMPONENTS, 3)).unbind(-1)
            weight = w.reshape(b, *([1] * (offset.ndim - 2)), COMPONENTS)
            return offset - (offset * weight).sum(-1, keepdim=True), 0.025 + F.softplus(scale), load

        pose_dist = distribution(self.position_dist(state_for_dist).unflatten(-1, (3, 3 * COMPONENTS)).flatten(-3, -2))
        reward_dist = distribution(self.reward_dist(rf_for_dist))

        fine = dict(
            pose=regular + shift[:, None, None, None],
            ref_pose=ref_pose + shift[:, None, None, None],
            edge=edges[:, :k],
            causal=causal[:, :k],
            strength=strength[:, :k],
            reward=self.reward_mean(reward_features)[..., 0],
            w=w,
            p_dist=pose_dist,
            r_dist=reward_dist,
            **facts,
        )
        if removed is not None:
            off = position[:, k + 1:].unflatten(1, (k, variants))
            fine['edge_effect'] = regular[:, :, None, ..., :3] - off[..., :3]

        # Aggregate native steps onto the 5 coarse comparison windows (matching Stage 29/30 interface)
        ends = d['end_idx'][0]
        active_w = d['window'][:, None, :, :, None, None]
        tw = d['window'] * d['native_widths'][:, None]
        tw = tw / tw.sum(-1, keepdim=True)
        out = dict(fine)
        for key in ('pose', 'ref_pose', 'causal'):
            out[key] = fine[key].index_select(2, ends)
        for key in ('edge', 'strength'):
            out[key] = fine[key][:, :, None].masked_fill(~active_w, -torch.inf).amax(3)
        out['reward'] = torch.einsum('bst,bkt->bks', tw, fine['reward'])
        a_d, s_d, l_d = fine['r_dist']
        out['r_dist'] = (
            torch.einsum('bst,bkth->bksh', tw, a_d),
            torch.einsum('bst,bkth->bksh', tw.square(), s_d.square()).sqrt(),
            torch.einsum('bst,bkth->bksh', tw, l_d),
        )
        out['p_dist'] = tuple(v.index_select(2, ends) for v in fine['p_dist'])
        if removed is not None:
            out['edge_effect'] = fine['edge_effect'].index_select(3, ends)
        if native:
            out['native'] = fine
        return out


def build_stabilized(kind, protocol):
    spec = protocol['variants'][kind.removesuffix('-calibrated')]
    return StabilizedWorld(
        mode=spec['mode'],
        hops=spec['hops'],
        width=spec['width'],
        feature=spec['feature'],
        gate_threshold=protocol['sparse_gate_threshold'],
    )
