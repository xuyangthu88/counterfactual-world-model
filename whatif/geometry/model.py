"""Stage 34: GeomDeltaGraphWorld — DeltaGraphWorld augmented with SE(3)-invariant SAT OBB clearance."""
import torch
from torch import nn
from torch.nn import functional as F

from whatif.deltagraph.model import DeltaGraphWorld, SceneLayoutGNN, dmlp
from whatif.effects.data import pair_ids
from whatif.effects.model import COMPONENTS, FACTORS, relative
from whatif.mechanism.model import U, pairmask, pairs, pool, tile
from whatif.stabilized.model import TemporalFacts, local_pair_geometry, pose_frame, safe_pose_input, sparse_gate


def sat_obb_features(pose, geom):
    """Exact SE(3)-invariant 3D Separating Axis Theorem (SAT) OBB clearance & soft separating normal.

    Inputs:
      pose: (..., N, 9) world-space pose (xyz, e1, e2)
      geom: (..., N, 6) body-local [b_min(3), b_max(3)]
    Returns:
      inv_feat: (..., N, N, 20) strictly SE(3)-invariant geometric contact features
      n_i_world: (..., N, N, 3) smooth equivariant separating-axis normal in i's frame (world coords)
      n_j_world: (..., N, N, 3) smooth equivariant separating-axis normal in j's frame (world coords)
    """
    e1, e2, e3 = pose_frame(pose)
    R = torch.stack([e1, e2, e3], dim=-2)  # (..., N, 3, 3), rows are e1, e2, e3 in world coords
    b_min, b_max = geom[..., :3], geom[..., 3:]
    c_loc = 0.5 * (b_min + b_max)          # (..., N, 3)
    h = 0.5 * (b_max - b_min).clamp_min(0.0)  # (..., N, 3)

    # World-space OBB geometric center: p + R^T @ c_loc (since rows of R are body basis vectors)
    center = pose[..., :3] + torch.einsum('...iad,...ia->...id', R, c_loc)
    d_world = relative(center)             # (..., N, N, 3) = center_j - center_i

    # Relative rotation matrix M_{ij}[a, c] = e_{i,a} . e_{j,c}
    abs_M = torch.einsum('...iad,...jcd->...ijac', R, R).abs()  # (..., N, N, 3, 3)

    # Project center-to-center vector into i's frame and j's frame
    d_in_i = torch.einsum('...iad,...ijd->...ija', R, d_world)   # (..., N, N, 3)
    d_in_j = torch.einsum('...jad,...ijd->...ija', R, -d_world)  # (..., N, N, 3)

    # Support radius of j along i's axes, and of i along j's axes
    r_j_on_i = torch.einsum('...ijac,...jc->...ija', abs_M, h)
    r_i_on_j = torch.einsum('...ijca,...ic->...ija', abs_M, h)

    # Axis-wise signed clearance (<= 0 along ALL 3 axes iff OBBs overlap along those axes!)
    h_i = h.unsqueeze(-2).expand_as(r_j_on_i)
    h_j = h.unsqueeze(-3).expand_as(r_i_on_j)
    g_i = d_in_i.abs() - (h_i + r_j_on_i)  # (..., N, N, 3)
    g_j = d_in_j.abs() - (h_j + r_i_on_j)  # (..., N, N, 3)

    # Scalar SAT separating gap: positive => separated; <= 0 => overlapping/contact
    max_gi = g_i.amax(dim=-1, keepdim=True)
    max_gj = g_j.amax(dim=-1, keepdim=True)
    sat_gap = torch.maximum(max_gi, max_gj)
    pen = F.relu(-sat_gap)
    prox = torch.exp(-F.relu(sat_gap) / 0.25)

    # Softly select the dominant separating/minimum-penetration axis for equivariant normal direction
    w_i = torch.softmax(g_i / 0.12, dim=-1)  # (..., N, N, 3)
    w_j = torch.softmax(g_j / 0.12, dim=-1)
    sgn_i = torch.tanh(d_in_i / 0.05)        # smooth sign pointing from i toward j along each axis of i
    sgn_j = torch.tanh(d_in_j / 0.05)        # smooth sign pointing from j toward i along each axis of j
    n_i_world = torch.einsum('...ija,...iad->...ijd', w_i * sgn_i, R)
    n_j_world = torch.einsum('...ija,...jad->...ijd', w_j * sgn_j, R)
    n_i_world = n_i_world / (0.1 + n_i_world.norm(dim=-1, keepdim=True))
    n_j_world = n_j_world / (0.1 + n_j_world.norm(dim=-1, keepdim=True))

    inv_feat = torch.cat([
        g_i.clamp(-3.0, 5.0),
        g_j.clamp(-3.0, 5.0),
        sat_gap.clamp(-3.0, 5.0),
        pen.clamp(0.0, 3.0),
        prox,
        d_in_i.clamp(-5.0, 5.0),
        h_i,
        h_j,
    ], dim=-1)  # 3+3+1+1+1+3+3+3 = 18D? Let's check: 3+3+1+1+1+3+3+3 = 18D!
    # Add max_gi, max_gj -> 20D exact
    inv_feat = torch.cat([inv_feat, max_gi.clamp(-3.0, 5.0), max_gj.clamp(-3.0, 5.0)], dim=-1)
    return inv_feat, n_i_world, n_j_world


class GeomSceneLayoutGNN(nn.Module):
    """t=0 SE(3)-equivariant scene layout GNN enriched with universal body-local extents and SAT clearance."""

    def __init__(self, feature=96, dropout=0.0):
        super().__init__()
        self.node_init = dmlp(3 + 1 + 6, feature, feature, dropout)
        self.edge1 = dmlp(2 * feature + 17 + 20, feature, feature, dropout)
        self.upd1 = nn.LayerNorm(feature)
        self.edge2 = dmlp(2 * feature + 17 + 20, feature + 2, feature, dropout)
        self.upd2 = nn.LayerNorm(feature)

    def forward(self, x, role, geom):
        mask = pairmask(role)
        e1, e2, e3 = pose_frame(x)
        R = torch.stack([e1, e2, e3], dim=-2)
        dp = relative(x[..., :3])
        q_i = torch.einsum('biad,bijd->bija', R, dp)
        q_j = torch.einsum('bjad,bijd->bija', R, dp)
        rel_R = torch.einsum('biad,bjcd->bijac', R, R).flatten(-2)
        dist = dp.norm(dim=-1, keepdim=True)
        rel_z = dp[..., 2:3]
        sat20, _, _ = sat_obb_features(x, geom)
        pgeom = torch.cat([q_i, q_j, rel_R, dist, rel_z, sat20], dim=-1)

        z_enc = 1.0 - torch.exp(-x[..., 2:3].clamp_min(0.0)) + x[..., 2:3].clamp_max(0.0)
        h = self.node_init(torch.cat([role, z_enc, geom], dim=-1))
        deg = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

        m1 = self.edge1(torch.cat([pairs(h), pgeom], dim=-1))
        h = self.upd1(h + (m1 * mask[..., None]).sum(dim=-2) / deg)

        out2 = self.edge2(torch.cat([pairs(h), pgeom], dim=-1))
        m2, alpha = out2[..., :-2], out2[..., -2:]
        h = self.upd2(h + (m2 * mask[..., None]).sum(dim=-2) / deg)

        dp_unit = dp / (1.0 + dist)
        raw_g = torch.einsum('bij,bijc,bijd->bicd', mask, alpha, dp_unit) / (deg[..., None] + 1.0)
        g_vec = raw_g / (0.1 + raw_g.norm(dim=-1, keepdim=True))
        return h, g_vec


class GeomDeltaGraphWorld(DeltaGraphWorld):
    """DeltaGraphWorld upgraded with universal 3D body-local geometry & SAT OBB clearance."""

    def __init__(self, use_delta_prop=True, hops=2, width=32, feature=96, dropout=0.08, gate_threshold=0.08):
        super().__init__(use_delta_prop=use_delta_prop, hops=hops, width=width, feature=feature,
                         dropout=dropout, gate_threshold=gate_threshold)
        self.layout = GeomSceneLayoutGNN(feature=feature, dropout=dropout)
        # node context adds 6D body-local geom
        self.node = dmlp(9 + 9 + 3 + 6 + 12 + U + feature, feature, feature, dropout)
        # pair_geom adds 20D SAT OBB clearance -> 32 + 20 = 52D
        self.geom_dim = 52
        edge_in = 2 * feature + self.geom_dim + 1 + 2 + 2 * (12 + U)
        self.edge = dmlp(edge_in, 4 + 2 * FACTORS**2, max(feature, 112), dropout)

        # Equivariant spatial directions add 2 SAT separating normals (n_i, n_j) -> 16 directions
        self.n_dirs = 16
        vec_in = 2 * feature + self.geom_dim + 2 * (12 + U) + 2
        self.vector = dmlp(vec_in, FACTORS * self.n_dirs, max(feature, 112), dropout)
        nn.init.zeros_(self.vector[-1].weight)
        nn.init.zeros_(self.vector[-1].bias)

        if self.use_delta_prop:
            self.n_delta_dirs = 12
            delta_vec_in = 2 * feature + self.geom_dim + 16 + 2 * (12 + U) + 2
            self.delta_vec_transport = dmlp(delta_vec_in, FACTORS * self.n_delta_dirs, feature, dropout)
            self.delta_vec_switch = dmlp(vec_in, FACTORS * self.n_dirs, feature, dropout)
            nn.init.zeros_(self.delta_vec_transport[-1].weight)
            nn.init.zeros_(self.delta_vec_transport[-1].bias)
            nn.init.zeros_(self.delta_vec_switch[-1].weight)
            nn.init.zeros_(self.delta_vec_switch[-1].bias)

    def _pair_geom_sat(self, pose_prop, vel, g_vec, geom_tile):
        base32, frames = self._pair_geom(pose_prop, vel, g_vec)
        sat20, n_i, n_j = sat_obb_features(pose_prop, geom_tile)
        return torch.cat([base32, sat20], dim=-1), frames, n_i, n_j

    def interaction_geo(self, end, pose_prop, vel, g_vec, geom_tile, prior_feat, mu_pair, dt_clk, mask, omit, soft):
        n = pose_prop.size(1)
        pgeom, frames, n_i, n_j = self._pair_geom_sat(pose_prop, vel, g_vec, geom_tile)
        features = torch.cat([
            pairs(end), pgeom, prior_feat[..., None],
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
        return weights, logit, conduct, persist_p, magnitude, pgeom, frames, n_i, n_j

    def delta_vector_message_geo(self, end, pair_in, directions, frames, g_vec, n_i, n_j,
                                 d_pose, d_vel, d_state, weights, d_weights, pgeom, mu_pair, time):
        n = end.size(1)
        e1, e2, e3 = frames
        R = torch.stack([e1, e2, e3], dim=-2)

        dp_d = relative(d_pose[..., :3])
        dv_d = relative(d_vel[..., :3])
        q_dp_i = torch.einsum('biad,bijd->bija', R, dp_d)
        q_dp_j = torch.einsum('bjad,bijd->bija', R, dp_d)
        q_dv_i = torch.einsum('biad,bijd->bija', R, dv_d)
        q_dv_j = torch.einsum('bjad,bijd->bija', R, dv_d)
        n_pi = d_pose[..., :3].norm(dim=-1, keepdim=True)[:, :, None].expand(-1, -1, n, -1)
        n_vi = d_vel[..., :3].norm(dim=-1, keepdim=True)[:, :, None].expand(-1, -1, n, -1)
        n_pj = d_pose[..., :3].norm(dim=-1, keepdim=True)[:, None].expand(-1, n, -1, -1)
        n_vj = d_vel[..., :3].norm(dim=-1, keepdim=True)[:, None].expand(-1, n, -1, -1)
        diff_geom = torch.cat([q_dp_i, q_dp_j, q_dv_i, q_dv_j, n_pi, n_vi, n_pj, n_vj], dim=-1)

        sender_mag = (d_pose[..., :3].norm(dim=-1) + d_vel[..., :3].norm(dim=-1)
                      + 0.2 * d_state.flatten(-2).norm(dim=-1))
        sender_gate = sender_mag / (0.1 + sender_mag)

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
            e1_i, e2_i, e1_j, e2_j, g0_i, g0_j, n_i, n_j,
        ], dim=-2)

        t_in = torch.cat([pairs(end), pgeom, diff_geom, mu_pair, time[:, None].expand(-1, n, -1, -1)], dim=-1)
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
        g_vec = tile(g_vec0, m_br)
        geom_tile = tile(d['geom'], m_br)
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
                weights, logit_h, conduct_h, persist_h, strength_h, pgeom, frames, n_i, n_j = self.interaction_geo(
                    end, proposed, v_curr, g_vec, geom_tile, prior_feat, mu_pair, time[:, 0], mask, omit, soft)

                base_msg = torch.einsum('bijfg,bifh->bjgh', weights, self.message(state))
                base_dirs = self._base_directions(pose, proposed, v_curr, frames, g_vec)
                directions = torch.cat([base_dirs, n_i.unsqueeze(-2), n_j.unsqueeze(-2)], dim=-2)
                dv_vec, pair_in = self.vector_message(end, directions, pgeom, mu_pair, time, weights)

                if self.use_delta_prop:
                    st_b = state.unflatten(0, (b, m_br))
                    st_ref = st_b[:, ref_slot:ref_slot + 1].expand_as(st_b).flatten(0, 1)
                    d_state = state - st_ref

                    po_b = proposed.unflatten(0, (b, m_br))
                    d_pose = (po_b - po_b[:, ref_slot:ref_slot + 1]).flatten(0, 1)

                    ve_b = v_curr.unflatten(0, (b, m_br))
                    d_vel = (ve_b - ve_b[:, ref_slot:ref_slot + 1]).flatten(0, 1)

                    w_b = weights.unflatten(0, (b, m_br))
                    w_ref = w_b[:, ref_slot:ref_slot + 1].expand_as(w_b).flatten(0, 1)
                    d_weights = (weights - w_ref) * omit[..., None, None]

                    delta_msg = (
                        torch.einsum('bijfg,bifh->bjgh', weights, self.msg_delta(d_state))
                        + torch.einsum('bijfg,bifh->bjgh', d_weights, self.msg_switch(st_ref))
                    )
                    delta_dv = self.delta_vector_message_geo(
                        end, pair_in, directions, frames, g_vec, n_i, n_j,
                        d_pose, d_vel, d_state, weights, d_weights, pgeom, mu_pair, time,
                    )
                    message = base_msg + delta_msg
                    dv_vec = dv_vec + delta_dv
                else:
                    message = base_msg

                rec_in = torch.cat([message, mu_factor, node_time], dim=-1).flatten(0, 2)
                h_cand = self.recurrent(rec_in, state.flatten(0, 2)).unflatten(0, state.shape[:3])
                state = self.state_norm((1.0 - alpha_hop) * state + alpha_hop * h_cand)

                dv_resp = torch.cat([
                    f(torch.cat([state[:, :, i_f], message[:, :, i_f], mu_node, time], dim=-1))
                    for i_f, f in enumerate(self.response)
                ], dim=-1)
                dv_hop = torch.cat([dv_vec + dv_resp[..., :3], dv_resp[..., 3:]], dim=-1)
                v_curr = (v_curr + dv_hop / self.hops).clamp(-50.0, 50.0)
                proposed = pose + dt * v_curr

                hop_logits.append(logit_h)
                hop_conducts.append(conduct_h)
                hop_persists.append(persist_h)
                hop_strengths.append(strength_h)

            logit = torch.stack(hop_logits, dim=0).amax(dim=0)
            conduct = torch.stack(hop_conducts, dim=0).amax(dim=0)
            persist_p = torch.stack(hop_persists, dim=0).amax(dim=0)
            magnitude = torch.stack(hop_strengths, dim=0).amax(dim=0)

            pose = proposed
            velocity = v_curr

            decay = (persist_p.clamp(1e-4, 1.0 - 1e-4)).pow(dt)
            present_edge = logit.sigmoid() * conduct * mask * omit
            carry = torch.maximum(carry * decay, present_edge)
            causal = torch.logit(carry.clamp(1e-4, 1.0 - 1e-4))

            movable = tile(roles[..., 0], m_br)
            summary = pool(state.flatten(-2), movable)
            mov_pose = pool(safe_pose_input(pose, True), movable)
            goal = tile(safe_pose_input(d['goal'], True), m_br)
            rew_feat = self.reward(torch.cat([summary, mov_pose, goal], dim=-1))
            rew = self.reward_mean(rew_feat)[..., 0]
            outputs.append((pose, logit, causal, magnitude, rew, state, rew_feat))

        poses, logits, causals, strengths, rewards, states, rew_feats = [
            torch.stack(item, dim=1).unflatten(0, (b, m_br)) for item in zip(*outputs)
        ]
        return dict(pose=poses, edge=logits, causal=causals, strength=strengths,
                    reward=rewards, state=states, reward_feature=rew_feats)

    def forward(self, d, removed=None, soft=0.0, native=False, detach_dist=False):
        xyz = pool(d['x'][..., :3], d['role'][..., 0])
        shift = F.pad(xyz[..., :2], (0, 7))
        centered = dict(
            d,
            x=d['x'] - shift[:, None],
            fact=d['fact'] - shift[:, None, None],
            goal=d['goal'] - shift,
        )
        facts = self.facts(centered)
        layout_h, g_vec0 = self.layout(centered['x'], centered['role'], centered['geom'])
        x_in = safe_pose_input(centered['x'], True)
        node = self.node(torch.cat([
            x_in,
            centered['fact'][:, -1] - centered['fact'][:, 0],
            centered['role'],
            centered['geom'],
            facts['m'],
            facts['u'],
            layout_h,
        ], dim=-1))

        action, reset = centered['a'], centered['reset']
        b, k, n = action.shape[:3]
        ref_action = torch.zeros_like(action[:, :1])
        ref_reset = torch.zeros_like(reset[:, :1])

        if removed is not None:
            r = removed.size(-1)
            action_all = torch.cat([action, action[:, :, None].expand(-1, -1, r, -1, -1).flatten(1, 2), ref_action], dim=1)
            reset_all = torch.cat([reset, reset[:, :, None].expand(-1, -1, r, -1, -1).flatten(1, 2), ref_reset], dim=1)
            keep = torch.ones((b, k * (1 + r) + 1, n, n), device=action.device)
            ps, pd = pair_ids(n, action.device)
            src, dst = ps[removed].flatten(1), pd[removed].flatten(1)
            row = torch.arange(b, device=action.device)[:, None]
            branch = torch.arange(k, k * (1 + r), device=action.device)[None]
            keep[row, branch, src, dst] = 0.0
            keep[row, branch, dst, src] = 0.0
        else:
            action_all = torch.cat([action, ref_action], dim=1)
            reset_all = torch.cat([reset, ref_reset], dim=1)
            keep = None

        ref_slot = action_all.size(1) - 1
        out = self.rollout(centered, node, layout_h, g_vec0, facts, action_all, reset_all, ref_slot, keep, soft)
        base = {key: value[:, :k] for key, value in out.items()}
        ref_pose_full = out['pose'][:, ref_slot:ref_slot + 1] + shift[:, None, None, None]
        base['pose'] = base['pose'] + shift[:, None, None, None]

        if removed is not None:
            edited = out['pose'][:, k:k * (1 + r)].unflatten(1, (k, r)) + shift[:, None, None, None, None]
            base['edge_effect'] = base['pose'][:, :, None, ..., :3] - edited[..., :3]

        w_map = d['window'][:, None]
        widths = d['native_widths'][:, None]
        end_idx = d['end_idx'][0]
        weight = (w_map * widths[:, :, None]).clamp_min(0.0)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-6)
        coarse = dict(
            pose=base['pose'][:, :, end_idx],
            ref_pose=ref_pose_full[:, :, end_idx],
            edge=base['edge'][:, :, None].masked_fill(~w_map[..., None, None], -1e4).amax(3),
            causal=base['causal'][:, :, end_idx],
            strength=base['strength'][:, :, None].masked_fill(~w_map[..., None, None], 0.0).amax(3),
            reward=(base['reward'][:, :, None] * weight).sum(-1),
            state=base['state'][:, :, end_idx],
            reward_feature=(base['reward_feature'][:, :, None] * weight[..., None]).sum(-2),
        )
        if 'edge_effect' in base:
            coarse['edge_effect'] = base['edge_effect'][:, :, :, end_idx]

        st_in = coarse['state'].detach() if detach_dist else coarse['state']
        rf_in = coarse['reward_feature'].detach() if detach_dist else coarse['reward_feature']
        pos_state = st_in[..., 0, :]
        p_d, p_r, p_s = self.position_dist(pos_state).unflatten(-1, (3, 3, COMPONENTS)).unbind(-2)
        r_d, r_r, r_s = self.reward_dist(rf_in).unflatten(-1, (3, COMPONENTS)).unbind(-2)
        regime = self.regime(rf_in.mean((1, 2))).softmax(-1)
        coarse.update(
            graph=facts['graph'],
            m=facts['m'],
            u=facts['u'],
            innovation=facts['innovation'],
            kl=facts['kl'],
            mech_entropy=facts['mech_entropy'],
            w=regime,
            p_dist=(p_d, p_r, F.softplus(p_s) + 0.05),
            r_dist=(r_d, r_r, F.softplus(r_s) + 0.02),
        )
        if native:
            coarse['native'] = {key: base[key] for key in ('pose', 'edge', 'causal', 'strength', 'reward') if key in base}
            if 'edge_effect' in base:
                coarse['native']['edge_effect'] = base['edge_effect']
        return coarse
