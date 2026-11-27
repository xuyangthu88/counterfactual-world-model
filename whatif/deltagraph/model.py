"""Stage 32: Equivariant Scene-Layout Geometry GNN + Direct Motion-Difference Graph Propagation (DeltaGraph)."""
import torch
from torch import nn
from torch.nn import functional as F

from whatif.effects.data import clock, pair_ids
from whatif.effects.model import COMPONENTS, FACTORS, average, relative
from whatif.mechanism.model import K, U, pairmask, pairs, pool, tile
from whatif.stabilized.model import (
    TemporalFacts,
    local_pair_geometry,
    pose_frame,
    safe_pose_input,
    sparse_gate,
)


def dmlp(i, o, h, dropout=0.0):
    layers = [nn.Linear(i, h), nn.SiLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(h, h), nn.SiLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(h, o))
    return nn.Sequential(*layers)


class SceneLayoutGNN(nn.Module):
    """Strictly t=0 rotation-equivariant geometric GNN over all valid scene objects.
    Computes per-object invariant layout embeddings and 2 equivariant 3D scene-geometry
    reference vectors (e.g., wall inward normals, tower overhang directions) BEFORE any
    action is injected, preserving strict graph causality and zero action leakage."""

    def __init__(self, feature=96, dropout=0.0):
        super().__init__()
        self.node_init = dmlp(3 + 1, feature, feature, dropout)
        self.edge1 = dmlp(2 * feature + 17, feature, feature, dropout)
        self.upd1 = nn.LayerNorm(feature)
        self.edge2 = dmlp(2 * feature + 17, feature + 2, feature, dropout)
        self.upd2 = nn.LayerNorm(feature)

    def forward(self, x, role):
        mask = pairmask(role)  # (B, N, N)
        e1, e2, e3 = pose_frame(x)
        R = torch.stack([e1, e2, e3], dim=-2)  # (B, N, 3, 3)
        dp = relative(x[..., :3])              # (B, N, N, 3) = x_j - x_i
        q_i = torch.einsum('biad,bijd->bija', R, dp)
        q_j = torch.einsum('bjad,bijd->bija', R, dp)
        rel_R = torch.einsum('biad,bjcd->bijac', R, R).flatten(-2)
        dist = dp.norm(dim=-1, keepdim=True)
        rel_z = dp[..., 2:3]
        geom = torch.cat([q_i, q_j, rel_R, dist, rel_z], dim=-1)  # 17D invariant

        z_enc = 1.0 - torch.exp(-x[..., 2:3].clamp_min(0.0)) + x[..., 2:3].clamp_max(0.0)
        h = self.node_init(torch.cat([role, z_enc], dim=-1))
        deg = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

        m1 = self.edge1(torch.cat([pairs(h), geom], dim=-1))
        agg1 = (m1 * mask[..., None]).sum(dim=-2) / deg
        h = self.upd1(h + agg1)

        out2 = self.edge2(torch.cat([pairs(h), geom], dim=-1))
        m2, alpha = out2[..., :-2], out2[..., -2:]  # alpha: (B, N, N, 2)
        agg2 = (m2 * mask[..., None]).sum(dim=-2) / deg
        h = self.upd2(h + agg2)

        # Equivariant 3D reference vectors: weighted sum of normalized (x_j - x_i)
        dp_unit = dp / (1.0 + dist)
        raw_g = torch.einsum('bij,bijc,bijd->bicd', mask, alpha, dp_unit) / (deg[..., None] + 1.0)
        g_vec = raw_g / (0.1 + raw_g.norm(dim=-1, keepdim=True))  # (B, N, 2, 3)
        return h, g_vec


class DeltaGraphWorld(nn.Module):
    """Shared counterfactual world model combining:
      1. Temporal kinematic physics abduction (16 mechanisms m, Gaussian u, revised factual graph).
      2. Strictly t=0 equivariant SceneLayoutGNN (inward boundary normals & structural overhang).
      3. Inertial-impulse velocity dynamics + SE(3) local-frame geometry + local PD actuator tracking.
      4. Shared within-step multi-hop constraint iterations (L=2 or 3).
      5. Direct Motion-Difference Graph Propagation (DeltaGraph): couples each candidate action
         to the scene's passive/factual reference rollout (a=0, reset=0), propagating state/motion
         differences along active edges W^a and topology-switch impulses along dW^a = (W^a - W^ref)*omit.
    """

    def __init__(self, use_delta_prop=True, hops=2, width=32, feature=96, dropout=0.08, gate_threshold=0.08):
        super().__init__()
        self.use_delta_prop = bool(use_delta_prop)
        self.hops = int(hops)
        self.width = width
        self.feature = feature
        self.gate_threshold = gate_threshold
        self.bound_z = True

        self.facts = TemporalFacts(feature=feature, bound_z=True)
        self.layout = SceneLayoutGNN(feature=feature, dropout=dropout)

        # node context: pose(9) + fact_disp(9) + role(3) + m(12) + u(4) + layout(F)
        self.node = dmlp(9 + 9 + 3 + 12 + U + feature, feature, feature, dropout)
        self.factor = nn.ModuleList([dmlp(feature + 3, width, feature, dropout) for _ in range(FACTORS)])
        self.action = dmlp(7 + 3 + 2, FACTORS * width, feature, dropout)

        # Endpoint encoder: state(3W) + msg(3W) + pose(9) + vel(9) + free(9) + m(12) + u(4) + layout(F)
        self.endpoint = dmlp(2 * FACTORS * width + 27 + 12 + U + feature, feature, feature, dropout)

        # Pairwise invariant geometry: 24D local_pair_geometry + 8D layout-normal projections = 32D
        self.geom_dim = 32
        edge_in = 2 * feature + self.geom_dim + 1 + 2 + 2 * (12 + U)
        self.edge = dmlp(edge_in, 4 + 2 * FACTORS**2, max(feature, 112), dropout)
        self.message = nn.Linear(width, width, bias=False)

        self.recurrent = nn.GRUCell(width + 12 + U + 2, width)
        self.state_norm = nn.LayerNorm(width)

        # Free inertial dynamics + local proportional command gain for ongoing actuators
        self.propose = nn.ModuleList([dmlp(width + 9 + 12 + U + 2, 3, feature, dropout) for _ in range(FACTORS)])
        self.retention = nn.ModuleList([dmlp(width + 12 + U + 2, 1, feature // 2, dropout) for _ in range(FACTORS)])
        for head in self.retention:
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, 0.5)
        self.cmd_gain = dmlp(width + 12 + U + 2, 1, feature // 2, dropout)
        nn.init.zeros_(self.cmd_gain[-1].weight)
        nn.init.constant_(self.cmd_gain[-1].bias, -0.5)

        self.response = nn.ModuleList([dmlp(2 * width + 12 + U + 2, 3, feature, dropout) for _ in range(FACTORS)])
        for obj in (*self.propose, *self.response):
            nn.init.normal_(obj[-1].weight, std=0.02)
            nn.init.zeros_(obj[-1].bias)

        # Equivariant spatial vector message (14 direction bases: 10 local-frame + 4 layout vectors)
        self.n_dirs = 14
        vec_in = 2 * feature + self.geom_dim + 2 * (12 + U) + 2
        self.vector = dmlp(vec_in, FACTORS * self.n_dirs, max(feature, 112), dropout)
        nn.init.zeros_(self.vector[-1].weight)
        nn.init.zeros_(self.vector[-1].bias)

        # Direct Motion-Difference Graph Propagation modules (zero-initialized for smooth training)
        if self.use_delta_prop:
            self.msg_delta = nn.Linear(width, width, bias=False)
            self.msg_switch = nn.Linear(width, width, bias=False)
            nn.init.zeros_(self.msg_delta.weight)
            nn.init.zeros_(self.msg_switch.weight)

            # Delta vector message: transport upstream (d_pose, d_vel) along active W^a,
            # and topology-switch impulse along dW^a = (W^a - W^ref) * omit.
            self.n_delta_dirs = 10
            # Invariant diff-pair features: q_dp_i(3)+q_dp_j(3)+q_dv_i(3)+q_dv_j(3)+norms(4) = 16D
            delta_vec_in = 2 * feature + self.geom_dim + 16 + 2 * (12 + U) + 2
            self.delta_vec_transport = dmlp(delta_vec_in, FACTORS * self.n_delta_dirs, feature, dropout)
            self.delta_vec_switch = dmlp(vec_in, FACTORS * self.n_dirs, feature, dropout)
            nn.init.zeros_(self.delta_vec_transport[-1].weight)
            nn.init.zeros_(self.delta_vec_transport[-1].bias)
            nn.init.zeros_(self.delta_vec_switch[-1].weight)
            nn.init.zeros_(self.delta_vec_switch[-1].bias)

        # Reward & coupled 4-component Gaussian mixture heads
        self.reward = dmlp(FACTORS * width + 18, feature, feature, dropout)
        self.reward_mean = nn.Linear(feature, 1)
        self.regime = dmlp(feature, COMPONENTS, feature, dropout)
        self.position_dist = dmlp(width, 3 * 3 * COMPONENTS, feature, dropout)
        self.reward_dist = nn.Linear(feature, 3 * COMPONENTS)

    def _pair_geom(self, pose_prop, vel, g_vec):
        base24, frames = local_pair_geometry(pose_prop, vel)
        dp = relative(pose_prop[..., :3])  # (BM, N, N, 3) = p_j - p_i
        dv = relative(vel[..., :3])
        # Project dp & dv onto equivariant layout vectors g_i, g_j (2 vectors each -> 8 scalars)
        g_i0, g_i1 = g_vec[:, :, 0], g_vec[:, :, 1]  # (BM, N, 3)
        p_i0 = torch.einsum('bid,bijd->bij', g_i0, dp)[..., None]
        p_i1 = torch.einsum('bid,bijd->bij', g_i1, dp)[..., None]
        p_j0 = torch.einsum('bjd,bijd->bij', g_i0, dp)[..., None]
        p_j1 = torch.einsum('bjd,bijd->bij', g_i1, dp)[..., None]
        v_i0 = torch.einsum('bid,bijd->bij', g_i0, dv)[..., None]
        v_i1 = torch.einsum('bid,bijd->bij', g_i1, dv)[..., None]
        v_j0 = torch.einsum('bjd,bijd->bij', g_i0, dv)[..., None]
        v_j1 = torch.einsum('bjd,bijd->bij', g_i1, dv)[..., None]
        layout_proj = torch.cat([p_i0, p_i1, p_j0, p_j1, v_i0, v_i1, v_j0, v_j1], dim=-1)
        return torch.cat([base24, layout_proj], dim=-1), frames

    def interaction(self, end, pose_prop, vel, g_vec, prior_feat, mu_pair, dt_clk, mask, omit, soft):
        n = pose_prop.size(1)
        geom, frames = self._pair_geom(pose_prop, vel, g_vec)
        features = torch.cat([
            pairs(end), geom, prior_feat[..., None],
            dt_clk[:, None, None].expand(-1, n, n, -1), mu_pair,
        ], dim=-1)
        p, c, strength, persist, opening, value = self.edge(features).split([1, 1, 1, 1, 9, 9], dim=-1)
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

    def _base_directions(self, pose, proposed, vel, frames, g_vec):
        n = pose.size(1)
        x = pose[..., :3]
        q = proposed[..., :3]
        v = vel[..., :3]
        e1, e2, e3 = frames

        def norm_v(z):
            return z / (1.0 + z.norm(dim=-1, keepdim=True))

        def endpoints(z):
            return z[:, :, None].expand(-1, -1, n, -1), z[:, None].expand(-1, n, -1, -1)

        e1_i, e1_j = endpoints(e1)
        e2_i, e2_j = endpoints(e2)
        _, e3_j = endpoints(e3)
        v_i, v_j = endpoints(v)
        g0_i, g0_j = endpoints(g_vec[:, :, 0])
        g1_i, g1_j = endpoints(g_vec[:, :, 1])
        return torch.stack([
            norm_v(relative(x)), norm_v(relative(q)),
            norm_v(v_i), norm_v(v_j), norm_v(relative(v)),
            e1_i, e2_i, e1_j, e2_j, e3_j,
            g0_i, g1_i, g0_j, g1_j,
        ], dim=-2)

    def vector_message(self, end, directions, geom, mu_pair, time, weights):
        n = end.size(1)
        pair_in = torch.cat([pairs(end), geom, mu_pair, time[:, None].expand(-1, n, -1, -1)], dim=-1)
        coef = self.vector(pair_in).unflatten(-1, (FACTORS, self.n_dirs))
        return torch.einsum('bijf,bijfl,bijlc->bjc', weights[..., 0], coef, directions), pair_in

    def delta_vector_message(self, end, pair_in, directions, frames, g_vec, d_pose, d_vel, d_state,
                             weights, d_weights, geom, mu_pair, time):
        """Propagate upstream motion differences along active edges W^a and edge-switch impulses along dW^a."""
        n = end.size(1)
        e1, e2, e3 = frames
        R = torch.stack([e1, e2, e3], dim=-2)

        dp_d = relative(d_pose[..., :3])  # d_pose_j - d_pose_i
        dv_d = relative(d_vel[..., :3])   # d_vel_j - d_vel_i
        q_dp_i = torch.einsum('biad,bijd->bija', R, dp_d)
        q_dp_j = torch.einsum('bjad,bijd->bija', R, dp_d)
        q_dv_i = torch.einsum('biad,bijd->bija', R, dv_d)
        q_dv_j = torch.einsum('bjad,bijd->bija', R, dv_d)
        n_pi = d_pose[..., :3].norm(dim=-1, keepdim=True)[:, :, None].expand(-1, -1, n, -1)
        n_vi = d_vel[..., :3].norm(dim=-1, keepdim=True)[:, :, None].expand(-1, -1, n, -1)
        n_pj = d_pose[..., :3].norm(dim=-1, keepdim=True)[:, None].expand(-1, n, -1, -1)
        n_vj = d_vel[..., :3].norm(dim=-1, keepdim=True)[:, None].expand(-1, n, -1, -1)
        diff_geom = torch.cat([q_dp_i, q_dp_j, q_dv_i, q_dv_j, n_pi, n_vi, n_pj, n_vj], dim=-1)

        # Gate transport term strictly by upstream sender difference magnitude (exact 0 if sender == ref!)
        sender_mag = (d_pose[..., :3].norm(dim=-1) + d_vel[..., :3].norm(dim=-1)
                      + 0.2 * d_state.flatten(-2).norm(dim=-1))
        sender_gate = sender_mag / (0.1 + sender_mag)  # (BM, N), exact 0 at 0!

        def norm_v(z):
            return z / (1.0 + z.norm(dim=-1, keepdim=True))

        def endpoints(z):
            return z[:, :, None].expand(-1, -1, n, -1), z[:, None].expand(-1, n, -1, -1)

        e1_i, e1_j = endpoints(e1)
        e2_i, e2_j = endpoints(e2)
        g0_i, g0_j = endpoints(g_vec[:, :, 0])
        dpi = d_pose[..., :3][:, :, None].expand(-1, -1, n, -1)
        dvi = d_vel[..., :3][:, :, None].expand(-1, -1, n, -1)
        delta_dirs = torch.stack([
            norm_v(dp_d), norm_v(dpi), norm_v(dv_d), norm_v(dvi),
            e1_i, e2_i, e1_j, e2_j, g0_i, g0_j,
        ], dim=-2)

        t_in = torch.cat([pairs(end), geom, diff_geom, mu_pair, time[:, None].expand(-1, n, -1, -1)], dim=-1)
        c_trans = self.delta_vec_transport(t_in).unflatten(-1, (FACTORS, self.n_delta_dirs))
        w_trans = weights[..., 0] * sender_gate[:, :, None, None]
        h_trans = torch.einsum('bijf,bijfl,bijlc->bjc', w_trans, c_trans, delta_dirs)

        c_sw = self.delta_vec_switch(pair_in).unflatten(-1, (FACTORS, self.n_dirs))
        h_sw = torch.einsum('bijf,bijfl,bijlc->bjc', d_weights[..., 0], c_sw, directions)
        return h_trans + h_sw

    def rollout(self, d, node, layout_h, g_vec0, facts, action, reset, ref_slot, keep=None, soft=0.0):
        x, roles = d['x'], d['role']
        b, m_br, n = action.shape[:3]
        x0 = tile(x, m_br)
        reset_flat = reset.flatten(0, 1)
        pose = x0 + reset_flat
        body = tile(node, m_br)
        lay = tile(layout_h, m_br)
        g_vec = tile(g_vec0, m_br)  # (BM, N, 2, 3)
        cmd = action.flatten(0, 1)
        mask = tile(pairmask(roles), m_br)
        omit = torch.ones_like(mask) if keep is None else keep.flatten(0, 1)

        prior_feat = torch.tanh(tile(facts['graph'], m_br) / 3.0)
        m_tile = tile(facts['m'], m_br)
        u_tile = tile(facts['u'], m_br)
        mu_node = torch.cat([m_tile, u_tile], dim=-1)
        mu_pair = pairs(mu_node)
        mu_factor = mu_node[:, :, None].expand(-1, -1, FACTORS, -1)

        ongoing = cmd[..., 0] * (reset_flat[..., :3].norm(dim=-1) < 1e-6).to(cmd)
        w1 = x0[..., :3] + cmd[..., 1:4]
        w2 = x0[..., :3] + cmd[..., 4:7]

        pose_init_in = safe_pose_input(pose, True)
        state = torch.stack([f(torch.cat([body, q], dim=-1))
                             for f, q in zip(self.factor, pose_init_in.split(3, dim=-1))], dim=-2)
        state = self.state_norm(state)
        message = torch.zeros_like(state)

        observed = (d['fact'][:, -1] - d['fact'][:, -2]) * (1.0 - d['same'][:, None, None])
        velocity = tile(observed / d['widths'][:, -1, None, None].clamp_min(1e-3), m_br)

        widths = d['native_widths']
        nominal = d['nominal_widths']
        elapsed = widths.cumsum(1) / widths.sum(1, keepdim=True)
        times = torch.stack([torch.log1p(nominal / 8.0), elapsed], dim=-1)
        carry = torch.zeros_like(prior_feat)

        outputs = []
        for t_clk, dt_raw, parent_raw in zip(times.unbind(1), widths.unbind(1), nominal.unbind(1)):
            time = tile(t_clk, m_br)[:, None].expand(-1, n, -1)
            dt = tile(dt_raw, m_br)[:, None, None]
            parent = tile(parent_raw, m_br)[:, None, None]
            fraction = (dt / parent).clamp(0.0, 1.0)
            s_prog = time[..., 1:2]

            s1 = (2.0 * s_prog).clamp(0.0, 1.0)
            s2 = (2.0 * s_prog - 1.0).clamp(0.0, 1.0)
            target = x0[..., :3] + s1 * (w1 - x0[..., :3]) + s2 * (w2 - w1)
            cmd_err = ongoing[..., None] * (target - pose[..., :3])

            source = self.action(torch.cat([cmd, cmd_err, time], dim=-1)).unflatten(-1, (FACTORS, self.width))
            state = self.state_norm(state + cmd[..., 0, None, None] * source * fraction[..., None])

            pose_in = safe_pose_input(pose, True)
            node_time = time[:, :, None].expand(-1, -1, FACTORS, -1)

            # Inertial free velocity + local PD proportional tracking for active ongoing actuators
            v_free_parts, free_acc_parts = [], []
            kp = F.softplus(self.cmd_gain(torch.cat([state[:, :, 0], mu_node, time], dim=-1)))
            for i_f, (f_acc, f_ret, p, vf) in enumerate(zip(
                    self.propose, self.retention, pose_in.split(3, -1), velocity.split(3, -1))):
                ret_base = f_ret(torch.cat([state[:, :, i_f], mu_node, time], dim=-1)).sigmoid()
                retain = ret_base.clamp(1e-4, 1.0 - 1e-4).pow(dt)
                acc = f_acc(torch.cat([state[:, :, i_f], p, vf, cmd_err, mu_node, time], dim=-1))
                v_f = retain * vf + dt * acc
                if i_f == 0:
                    v_f = v_f + ongoing[..., None] * kp * cmd_err
                v_free_parts.append(v_f)
                free_acc_parts.append(acc)
            v_curr = torch.cat(v_free_parts, dim=-1).clamp(-50.0, 50.0)
            free_acc = torch.cat(free_acc_parts, dim=-1)
            proposed = pose + dt * v_curr

            hop_logits, hop_conducts, hop_persists, hop_strengths = [], [], [], []
            alpha_hop = (fraction / self.hops)[..., None]

            for _ in range(self.hops):
                prop_in = safe_pose_input(proposed, True)
                end = self.endpoint(torch.cat([
                    state.flatten(-2), message.flatten(-2), prop_in, v_curr, free_acc, mu_node, lay,
                ], dim=-1))
                weights, logit_h, conduct_h, persist_h, strength_h, geom, frames = self.interaction(
                    end, proposed, v_curr, g_vec, prior_feat, mu_pair, time[:, 0], mask, omit, soft)

                base_msg = torch.einsum('bijfg,bifh->bjgh', weights, self.message(state))
                directions = self._base_directions(pose, proposed, v_curr, frames, g_vec)
                dv_vec, pair_in = self.vector_message(end, directions, geom, mu_pair, time, weights)

                if self.use_delta_prop:
                    # Reference-contrast differences strictly within each scene (ref_slot = passive/factual ref)
                    st_b = state.unflatten(0, (b, m_br))
                    st_ref = st_b[:, ref_slot:ref_slot + 1].expand_as(st_b).flatten(0, 1)
                    d_state = state - st_ref

                    po_b = proposed.unflatten(0, (b, m_br))
                    d_pose = (po_b - po_b[:, ref_slot:ref_slot + 1]).flatten(0, 1)

                    ve_b = v_curr.unflatten(0, (b, m_br))
                    d_vel = (ve_b - ve_b[:, ref_slot:ref_slot + 1]).flatten(0, 1)

                    w_b = weights.unflatten(0, (b, m_br))
                    w_ref = w_b[:, ref_slot:ref_slot + 1].expand_as(w_b).flatten(0, 1)
                    # Mask d_W by omit so ZERO message ever crosses a disabled edge B_ij = 0!
                    d_weights = (weights - w_ref) * omit[..., None, None]

                    delta_msg = (
                        torch.einsum('bijfg,bifh->bjgh', weights, self.msg_delta(d_state))
                        + torch.einsum('bijfg,bifh->bjgh', d_weights, self.msg_switch(st_ref))
                    )
                    delta_dv = self.delta_vector_message(
                        end, pair_in, directions, frames, g_vec, d_pose, d_vel, d_state,
                        weights, d_weights, geom, mu_pair, time,
                    )
                    message = base_msg + delta_msg
                    dv_vec = dv_vec + delta_dv
                else:
                    message = base_msg

                gru_in = torch.cat([message, mu_factor, node_time], dim=-1).flatten(0, 2)
                next_h = self.recurrent(gru_in, state.flatten(0, 2)).reshape_as(state)
                state = self.state_norm((1.0 - alpha_hop) * state + alpha_hop * next_h)

                dv_parts = []
                for i_f, f in enumerate(self.response):
                    h_f, m_f = state[:, :, i_f], message[:, :, i_f]
                    base_ctx = torch.cat([h_f, torch.zeros_like(m_f), mu_node, time], dim=-1)
                    full_ctx = torch.cat([h_f, m_f, mu_node, time], dim=-1)
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
        return [torch.stack(v, dim=1).unflatten(0, (b, m_br)) for v in zip(*outputs)]

    def forward(self, d, removed=None, *, soft=0.0, native=False, detach_dist=True):
        xyz = pool(d['x'][..., :3], d['role'][..., 0])
        shift = F.pad(xyz[..., :2], (0, 7))
        cd = dict(d, x=d['x'] - shift[:, None], fact=d['fact'] - shift[:, None, None], goal=d['goal'] - shift)

        facts = self.facts(cd)
        x, role, action = cd['x'], cd['role'], cd['a']
        layout_h, g_vec0 = self.layout(x, role)
        node = self.node(torch.cat([
            safe_pose_input(x, True), cd['fact'][:, -1] - cd['fact'][:, 0], role, facts['m'], facts['u'], layout_h,
        ], dim=-1))

        # Universal passive/factual reference branch: a=0, reset=0 (placed at slot k)
        ref_act = torch.zeros_like(action[:, :1])
        ref_reset = torch.zeros_like(cd['reset'][:, :1])
        all_action = torch.cat([action, ref_act], dim=1)
        all_reset = torch.cat([cd['reset'], ref_reset], dim=1)
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

        position, edges, causal, strength, state = self.rollout(
            cd, node, layout_h, g_vec0, facts, all_action, all_reset, ref_slot=k, keep=keep, soft=soft,
        )
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


def build_deltagraph(kind, protocol):
    spec = protocol['variants'][kind.removesuffix('-calibrated')]
    return DeltaGraphWorld(
        use_delta_prop=spec['use_delta_prop'],
        hops=spec['hops'],
        width=spec['width'],
        feature=spec['feature'],
        dropout=spec['dropout'],
        gate_threshold=protocol['sparse_gate_threshold'],
    )
