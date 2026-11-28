"""Stage 35: MultiConvexDeltaGraphWorld — Universal M=2 multi-convex SAT-OBB + contact lever arm &
equivariant SO(3) torque/angular impulse + intervened-state layout GNN + SAT-equipped TemporalFacts +
Pearl divergence-gated step-wise exogenous abduction for same-start counterfactual worlds."""
import torch
from torch import nn
from torch.nn import functional as F

from whatif.deltagraph.model import DeltaGraphWorld, dmlp
from whatif.effects.data import clock, pair_ids
from whatif.effects.model import COMPONENTS, FACTORS, average, relative
from whatif.mechanism.model import K, U, mlp, pairmask, pairs, pool, tile
from whatif.stabilized.model import pose_frame, safe_pose_input, sparse_gate


def multi_sat_features(pose, parts, part_mask):
    """Exact SE(3)-invariant 3D Multi-Convex (M<=2) SAT clearance, contact lever arm, and torque axes.

    Inputs:
      pose:      (..., N, 9) world pose [xyz, r1, r2]
      parts:     (..., N, M, 6) body-local convex sub-boxes [b_min(3), b_max(3)]
      part_mask: (..., N, M) binary validity mask for each sub-part
    Returns:
      inv_feat:    (..., N, N, 32) strictly SE(3)-invariant geometric contact + lever + torque features
      n_i_world:   (..., N, N, 3) equivariant separating normal from i's frame (world coords)
      n_j_world:   (..., N, N, 3) equivariant separating normal from j's frame (world coords)
      r_i_world:   (..., N, N, 3) equivariant contact lever arm on body i (world coords)
      r_j_world:   (..., N, N, 3) equivariant contact lever arm on body j (world coords)
      tau_i_world: (..., N, N, 3) equivariant unit geometric torque axis on body i (r_i x n_j)
      tau_j_world: (..., N, N, 3) equivariant unit geometric torque axis on body j (r_j x n_i)
    """
    e1, e2, e3 = pose_frame(pose)
    R = torch.stack([e1, e2, e3], dim=-2)  # (..., N, 3, 3), rows are e1, e2, e3

    b_min, b_max = parts[..., :3], parts[..., 3:]      # (..., N, M, 3)
    c_loc = 0.5 * (b_min + b_max)                      # (..., N, M, 3)
    h = 0.5 * (b_max - b_min).clamp_min(0.0)           # (..., N, M, 3)

    # Universal volume-weighted body-local CoM
    vol = (2.0 * h).clamp_min(1e-3).prod(dim=-1) * part_mask  # (..., N, M)
    com_loc = (c_loc * vol[..., None]).sum(dim=-2) / vol.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    # World-space centers of every sub-part: p + R^T @ c_loc
    part_center = pose[..., None, :3] + torch.einsum('...iad,...ima->...imd', R, c_loc)  # (..., N, M, 3)
    d_world = part_center[..., None, :, None, :, :] - part_center[..., :, None, :, None, :]  # (..., N, N, M, M, 3)

    # Shared relative rotation matrix |R_i R_j^T| across all sub-parts of (i, j)
    abs_M = torch.einsum('...iad,...jcd->...ijac', R, R).abs()  # (..., N, N, 3, 3)
    d_in_i = torch.einsum('...iad,...ijpqd->...ijpqa', R, d_world)
    d_in_j = torch.einsum('...jad,...ijpqd->...ijpqa', R, -d_world)

    r_j_on_i = torch.einsum('...ijac,...jqc->...ijqa', abs_M, h).unsqueeze(-3)  # (..., N, N, 1, M, 3)
    r_i_on_j = torch.einsum('...ijca,...ipc->...ijpa', abs_M, h).unsqueeze(-2)  # (..., N, N, M, 1, 3)
    h_i_exp = h[..., :, None, :, None, :]  # (..., N, 1, M, 1, 3)
    h_j_exp = h[..., None, :, None, :, :]  # (..., 1, N, 1, M, 3)

    g_i = d_in_i.abs() - (h_i_exp + r_j_on_i)  # (..., N, N, M, M, 3)
    g_j = d_in_j.abs() - (h_j_exp + r_i_on_j)  # (..., N, N, M, M, 3)
    sat_pq = torch.maximum(g_i.amax(dim=-1), g_j.amax(dim=-1))  # (..., N, N, M, M)

    valid_pq = (part_mask[..., :, None, :, None] * part_mask[..., None, :, None, :]).flatten(-2)  # (..., N, N, M*M)
    sat_flat = sat_pq.flatten(-2)
    sat_masked = torch.where(valid_pq > 0.5, sat_flat, torch.full_like(sat_flat, 1e4))

    sat_sorted, _ = sat_masked.sort(dim=-1)
    sat_min = sat_sorted[..., 0:1].clamp(-3.0, 5.0)
    has_2nd = valid_pq.sum(dim=-1, keepdim=True) > 1.5
    sat_2nd = torch.where(has_2nd, sat_sorted[..., 1:2], sat_sorted[..., 0:1]).clamp(-3.0, 5.0)

    logits_pq = torch.where(valid_pq > 0.5, -sat_flat / 0.06, torch.full_like(sat_flat, -1e4))
    alpha_pq = torch.softmax(logits_pq, dim=-1)  # (..., N, N, M*M)

    g_i_f = g_i.flatten(-3, -2)
    g_j_f = g_j.flatten(-3, -2)
    d_i_f = d_in_i.flatten(-3, -2)
    d_j_f = d_in_j.flatten(-3, -2)
    h_i_f = h_i_exp.expand_as(g_i).flatten(-3, -2)
    h_j_f = h_j_exp.expand_as(g_j).flatten(-3, -2)

    g_i_w = (alpha_pq[..., None] * g_i_f).sum(dim=-2)
    g_j_w = (alpha_pq[..., None] * g_j_f).sum(dim=-2)
    d_i_w = (alpha_pq[..., None] * d_i_f).sum(dim=-2)
    h_i_w = (alpha_pq[..., None] * h_i_f).sum(dim=-2)
    h_j_w = (alpha_pq[..., None] * h_j_f).sum(dim=-2)

    # Smooth equivariant separating normals from the winning sub-part pair
    w_ax_i = torch.softmax(g_i_f / 0.12, dim=-1)
    w_ax_j = torch.softmax(g_j_f / 0.12, dim=-1)
    n_i_loc = (alpha_pq[..., None] * (w_ax_i * torch.tanh(d_i_f / 0.05))).sum(dim=-2)
    n_j_loc = (alpha_pq[..., None] * (w_ax_j * torch.tanh(d_j_f / 0.05))).sum(dim=-2)
    n_i_world = torch.einsum('...ija,...iad->...ijd', n_i_loc, R)
    n_j_world = torch.einsum('...ija,...jad->...ijd', n_j_loc, R)
    n_i_world = n_i_world / (0.1 + n_i_world.norm(dim=-1, keepdim=True))
    n_j_world = n_j_world / (0.1 + n_j_world.norm(dim=-1, keepdim=True))

    # Closest-point contact lever arms relative to each body's CoM
    surf_i_loc = c_loc[..., :, None, :, None, :] + torch.maximum(torch.minimum(d_in_i, h_i_exp), -h_i_exp)
    surf_j_loc = c_loc[..., None, :, None, :, :] + torch.maximum(torch.minimum(d_in_j, h_j_exp), -h_j_exp)
    lever_i_loc = (alpha_pq[..., None] * (surf_i_loc - com_loc[..., :, None, None, None, :]).flatten(-3, -2)).sum(dim=-2)
    lever_j_loc = (alpha_pq[..., None] * (surf_j_loc - com_loc[..., None, :, None, None, :]).flatten(-3, -2)).sum(dim=-2)
    r_i_world = torch.einsum('...ija,...iad->...ijd', lever_i_loc, R)
    r_j_world = torch.einsum('...ija,...jad->...ijd', lever_j_loc, R)

    # Unit geometric torque axes (r x n) induced by normal contact impulse
    tau_j_world = torch.linalg.cross(r_j_world, n_i_world, dim=-1)
    tau_i_world = torch.linalg.cross(r_i_world, n_j_world, dim=-1)
    tau_j_loc = torch.einsum('...jad,...ijd->...ija', R, tau_j_world)
    tau_i_loc = torch.einsum('...iad,...ijd->...ija', R, tau_i_world)

    pen = F.relu(-sat_min)
    prox_sharp = torch.exp(-F.relu(sat_min) / 0.12)
    prox_wide = torch.exp(-F.relu(sat_min) / 0.35)

    inv_feat = torch.cat([
        g_i_w.clamp(-3.0, 5.0),          # 3
        g_j_w.clamp(-3.0, 5.0),          # 3
        sat_min,                         # 1
        sat_2nd,                         # 1
        pen.clamp(0.0, 3.0),             # 1
        prox_sharp,                      # 1
        prox_wide,                       # 1
        d_i_w.clamp(-5.0, 5.0),          # 3
        lever_i_loc.clamp(-3.0, 3.0),    # 3
        lever_j_loc.clamp(-3.0, 3.0),    # 3
        tau_i_loc.clamp(-3.0, 3.0),      # 3
        tau_j_loc.clamp(-3.0, 3.0),      # 3
        h_i_w,                           # 3
        h_j_w,                           # 3
    ], dim=-1)                           # total = 32D exact
    return inv_feat, n_i_world, n_j_world, r_i_world, r_j_world, tau_i_world, tau_j_world


def interpolate_factual_native(fact, coarse_widths, native_widths):
    """Linearly interpolate coarse factual trajectory (B, T_f, N, 9) onto native timestamps (B, T_n+1, N, 9).

    Uses ONLY factual observations & step timestamps (zero action/reset dependence).
    """
    b, t_f, n, d = fact.shape
    t_n = native_widths.size(1)
    dev = fact.device
    tau_f = F.pad(coarse_widths.cumsum(dim=1), (1, 0))           # (B, T_f)
    tau_n = F.pad(native_widths.cumsum(dim=1), (1, 0))           # (B, T_n + 1)

    # Locate right bracket index for each native timestamp per batch item
    # For uniform batch timestamps this vectorizes cleanly via searchsorted on contiguous tensors
    idx_hi = torch.searchsorted(tau_f.contiguous(), tau_n.contiguous(), right=False).clamp(1, t_f - 1)
    idx_lo = idx_hi - 1
    row = torch.arange(b, device=dev)[:, None]
    t0 = tau_f[row, idx_lo]
    t1 = tau_f[row, idx_hi]
    frac = ((tau_n - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0.0, 1.0)[..., None, None]
    f0 = fact[row, idx_lo]
    f1 = fact[row, idx_hi]
    return f0 + frac * (f1 - f0)


class MultiConvexTemporalFacts(nn.Module):
    """Physics abduction enriched with Multi-Convex SAT contact clearance & continuous decay-rate kinematics."""

    def __init__(self, feature=96, bound_z=True):
        super().__init__()
        self.feature = feature
        self.bound_z = bound_z
        # pose(9) + cmd(7) + role(3) + geom_summary(12) + clock(2) + vel(9) + acc(9) + spd(1) + ratio(1)
        # + damp_rate(1) + omega_z(1) + min_sat(1) + contact_prox(1) = 57
        self.local = mlp(57, feature, feature)
        self.prior = mlp(2 * feature + 4 + 32, 1, feature)
        self.motion = mlp(2 * feature + 9 + 1, 9, feature)
        self.step_enc = mlp(feature + 27 + 2, feature, feature)
        self.temporal = nn.GRU(feature, feature, batch_first=True)
        self.history_norm = nn.LayerNorm(feature)
        self.attend = nn.Linear(feature, 1)
        self.latent = mlp(feature + 1, K + 2 * U, feature)
        self.mechanisms = nn.Parameter(torch.randn(K, 12) * 0.25)
        self.correct = mlp(2 * (feature + 12 + U + 9) + 1 + 32, 1, feature)
        self.refine = mlp(feature + 2, K + 2 * U, feature)

    def forward(self, d):
        role, factual = d['role'], d['fact']
        parts, part_mask = d['parts'], d['part_mask']
        current, next_pose = factual[:, :-1], factual[:, 1:]
        b, t, n, _ = current.shape
        dt = d['widths'][:, :, None, None].clamp_min(1e-3)
        vel = (next_pose - current) / dt
        prev_vel = torch.cat([torch.zeros_like(vel[:, :1]), vel[:, :-1]], dim=1)
        acc = vel - prev_vel
        speed = vel[..., :3].norm(dim=-1, keepdim=True)
        prev_speed = prev_vel[..., :3].norm(dim=-1, keepdim=True)
        ratio = (speed / (prev_speed + 0.05)).clamp(0.0, 4.0)
        # Physics-invariant continuous exponential decay rate -log(v_t / v_{t-1}) / dt (directly measures damping/drag!)
        moving_prev = (prev_speed > 0.03).to(speed)
        damp_rate = (moving_prev * (-torch.log((speed + 0.02) / (prev_speed + 0.02)) / dt)).clamp(-2.0, 4.0)
        # Planar angular velocity estimate from rotation basis derivative e1_dot . e2
        e1_c, e2_c, _ = pose_frame(current)
        omega_z = (vel[..., 3:6] * e2_c).sum(dim=-1, keepdim=True).clamp(-10.0, 10.0)

        # Multi-convex SAT at every factual frame so abduction distinguishes wall contact from free damping!
        parts_t = parts[:, None].expand(-1, t, -1, -1, -1)
        pmask_t = part_mask[:, None].expand(-1, t, -1, -1)
        sat32_t, *_ = multi_sat_features(current, parts_t, pmask_t)
        valid = pairmask(role)  # (B, N, N)
        # Per-node minimum SAT clearance to any other present object at frame t
        sat_gap_t = sat32_t[..., 6].masked_fill(valid[:, None] < 0.5, 5.0)
        min_sat_node = sat_gap_t.amin(dim=-1, keepdim=True).clamp(-3.0, 5.0)
        contact_prox_node = torch.exp(-F.relu(min_sat_node) / 0.15)

        command = d['fa'][:, None].expand(-1, t, -1, -1)
        pose_in = safe_pose_input(current, self.bound_z)
        role_t = role[:, None].expand(-1, t, -1, -1)
        geom12 = parts.flatten(-2)[:, None].expand(-1, t, -1, -1)
        clk_t = clock(d['widths'])[:, :, None].expand(-1, -1, n, -1)
        local = self.local(torch.cat([
            pose_in, command, role_t, geom12, clk_t, vel, acc, speed, ratio,
            damp_rate, omega_z, min_sat_node, contact_prox_node,
        ], dim=-1))

        rel_p = relative(current[..., :3])
        rel_d = rel_p.norm(dim=-1, keepdim=True)
        prior = self.prior(torch.cat([pairs(local), rel_p, rel_d, sat32_t], dim=-1))[..., 0]
        prob = prior.sigmoid() * valid[:, None]
        message = torch.einsum('btij,btih->btjh', prob, local)
        motion = self.motion(torch.cat([
            local, message / prob.sum(-2).clamp_min(1.0)[..., None], vel, contact_prox_node,
        ], dim=-1))
        innovation = next_pose - current - motion

        enc = self.step_enc(torch.cat([local, vel, motion, innovation, damp_rate, contact_prox_node], dim=-1))
        seq = enc.transpose(1, 2).reshape(b * n, t, self.feature)
        _, h_last = self.temporal(seq)
        history = self.history_norm(h_last.squeeze(0).reshape(b, n, self.feature))

        alpha_t = self.attend(enc).softmax(dim=1)
        innovation_avg = (innovation * alpha_t).sum(dim=1)
        alpha_scalar = alpha_t[..., 0]
        w_pair = (alpha_scalar[..., None] + alpha_scalar[:, :, None, :]) * 0.5
        initial = (prior * w_pair).sum(dim=1)
        initial = 0.5 * (initial + initial.transpose(-1, -2))
        sat32_avg = (sat32_t * w_pair[..., None]).sum(dim=1)

        def degree(logit):
            return (logit.sigmoid() * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1.0)

        def posterior(v):
            coef, mean, log_sd = v.split([K, U, U], dim=-1)
            probs = coef.softmax(-1)
            return probs @ self.mechanisms, mean, log_sd.clamp(-4.0, 1.0), probs

        first = self.latent(torch.cat([history, degree(initial)], dim=-1))
        m0, u0, _, _ = posterior(first)
        edit = torch.cat([
            pairs(torch.cat([history, m0, u0, innovation_avg], dim=-1)),
            initial[..., None],
            sat32_avg,
        ], dim=-1)
        factual_graph = initial + self.correct(edit)[..., 0]
        factual_graph = 0.5 * (factual_graph + factual_graph.transpose(-1, -2))

        refined = first + self.refine(torch.cat([history, degree(initial), degree(factual_graph)], dim=-1))
        m, mean, scale, probs = posterior(refined)
        eps = d.get('eps')
        if eps is None:
            eps = torch.randn_like(mean) if self.training else torch.zeros_like(mean)
        kl = 0.5 * (mean.square() + (2.0 * scale).exp() - 1.0 - 2.0 * scale)

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


class MultiConvexLayoutGNN(nn.Module):
    """Strictly t=0 SE(3)-equivariant scene layout GNN operating on multi-convex SAT primitives.

    Accepts explicit edge_mask (= pairmask * omit) so intervened t=0 states (x0 + reset) get
    their true intervened support/overhang layout while guaranteeing 0.00e+00 action/reset leakage
    when edges are masked!
    """

    def __init__(self, feature=96, dropout=0.0):
        super().__init__()
        # role(3) + z_enc(1) + parts(12) + part_mask(2) = 18
        self.node_init = dmlp(18, feature, feature, dropout)
        # 2*F + base17 + sat32 = 2*F + 49
        self.edge1 = dmlp(2 * feature + 49, feature, feature, dropout)
        self.upd1 = nn.LayerNorm(feature)
        self.edge2 = dmlp(2 * feature + 49, feature + 2, feature, dropout)
        self.upd2 = nn.LayerNorm(feature)

    def forward(self, x, role, parts, part_mask, edge_mask=None):
        mask = pairmask(role) if edge_mask is None else edge_mask
        e1, e2, e3 = pose_frame(x)
        R = torch.stack([e1, e2, e3], dim=-2)
        dp = relative(x[..., :3])
        q_i = torch.einsum('biad,bijd->bija', R, dp)
        q_j = torch.einsum('bjad,bijd->bija', R, dp)
        rel_R = torch.einsum('biad,bjcd->bijac', R, R).flatten(-2)
        dist = dp.norm(dim=-1, keepdim=True)
        rel_z = dp[..., 2:3]
        sat32, *_ = multi_sat_features(x, parts, part_mask)
        pgeom = torch.cat([q_i, q_j, rel_R, dist, rel_z, sat32], dim=-1)

        z_enc = 1.0 - torch.exp(-x[..., 2:3].clamp_min(0.0)) + x[..., 2:3].clamp_max(0.0)
        h = self.node_init(torch.cat([role, z_enc, parts.flatten(-2), part_mask], dim=-1))
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


class MultiConvexDeltaGraphWorld(DeltaGraphWorld):
    """Stage 35 shared counterfactual world model:
      1. Universal M=2 Multi-Convex SAT-OBB clearance, contact lever arms, and unit torque axes.
      2. SAT-equipped TemporalFacts with physics-invariant exponential decay-rate kinematics.
      3. Intervened-state MultiConvexLayoutGNN (evaluates true intervened x0 + reset tower/scene
         with explicit edge_mask = mask * omit for strict 0.00e+00 leakage).
      4. Equivariant linear + angular (torque -> omega x e1, omega x e2) graph impulses & DeltaGraph.
      5. Optional Pearl divergence-gated step-wise exogenous abduction for same-start worlds (same == 1).
    """

    def __init__(self, use_delta_prop=True, use_exo=True, hops=3, width=40, feature=128,
                 dropout=0.10, gate_threshold=0.08):
        super().__init__(use_delta_prop=use_delta_prop, hops=hops, width=width, feature=feature,
                         dropout=dropout, gate_threshold=gate_threshold)
        self.use_exo = bool(use_exo)
        self.facts = MultiConvexTemporalFacts(feature=feature, bound_z=True)
        self.layout = MultiConvexLayoutGNN(feature=feature, dropout=dropout)

        # node context adds 12D flattened parts + 2D part_mask (instead of 6D single geom)
        self.node = dmlp(9 + 9 + 3 + 14 + 12 + U + feature, feature, feature, dropout)

        # Pairwise invariant geometry: 32D base + 32D multi-convex SAT/lever/torque = 64D
        self.geom_dim = 64
        edge_in = 2 * feature + self.geom_dim + 1 + 2 + 2 * (12 + U)
        self.edge = dmlp(edge_in, 4 + 2 * FACTORS**2, max(feature, 128), dropout)

        # Linear equivariant directions: 14 base + n_i + n_j + r_j_world = 17 directions
        self.n_dirs = 17
        vec_in = 2 * feature + self.geom_dim + 2 * (12 + U) + 2
        self.vector = dmlp(vec_in, FACTORS * self.n_dirs, max(feature, 128), dropout)
        nn.init.zeros_(self.vector[-1].weight)
        nn.init.zeros_(self.vector[-1].bias)

        # Equivariant angular impulse directions (for rigid-body rotation omega x e1, omega x e2):
        # [tau_j, r_j x dv_rel, r_j x dp_unit, tau_i, e1_j, e2_j, e3_j] = 7 equivariant pseudovectors
        self.n_ang_dirs = 7
        self.ang_vector = dmlp(vec_in, FACTORS * self.n_ang_dirs, max(feature, 96), dropout)
        nn.init.zeros_(self.ang_vector[-1].weight)
        nn.init.zeros_(self.ang_vector[-1].bias)

        if self.use_delta_prop:
            # Delta linear directions: 10 base + n_i + n_j + r_j = 13
            self.n_delta_dirs = 13
            delta_vec_in = 2 * feature + self.geom_dim + 16 + 2 * (12 + U) + 2
            self.delta_vec_transport = dmlp(delta_vec_in, FACTORS * self.n_delta_dirs, feature, dropout)
            self.delta_vec_switch = dmlp(vec_in, FACTORS * self.n_dirs, feature, dropout)
            # Delta angular impulse transport & switch (6 torque pseudovector bases)
            self.n_delta_ang_dirs = 6
            self.delta_ang_transport = dmlp(delta_vec_in, FACTORS * self.n_delta_ang_dirs, feature, dropout)
            self.delta_ang_switch = dmlp(vec_in, FACTORS * self.n_ang_dirs, feature, dropout)
            for head in (self.delta_vec_transport, self.delta_vec_switch,
                         self.delta_ang_transport, self.delta_ang_switch):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

        if self.use_exo:
            # Divergence-gated Pearl step-wise exogenous abduction residual for same == 1 worlds
            # Inputs: [||dp||, ||dv||, ||dW_in||_1, ||reset||, has_omit] = 5D strictly invariant scalars
            self.exo_gate = nn.Sequential(
                nn.Linear(5 + 12 + U, 32),
                nn.SiLU(),
                nn.Linear(32, 2),  # [pos_gate_logit, rot_gate_logit]
            )
            nn.init.zeros_(self.exo_gate[-1].weight)
            nn.init.constant_(self.exo_gate[-1].bias, -1.5)  # start conservatively near 0.18, learn cleanly
            self.exo_log_decay = nn.Parameter(torch.tensor(0.5))

    def _pair_geom_mc(self, pose_prop, vel, g_vec, parts_tile, pmask_tile):
        base32, frames = self._pair_geom(pose_prop, vel, g_vec)
        sat32, n_i, n_j, r_i, r_j, tau_i, tau_j = multi_sat_features(pose_prop, parts_tile, pmask_tile)
        return torch.cat([base32, sat32], dim=-1), frames, (n_i, n_j, r_i, r_j, tau_i, tau_j)

    def interaction_mc(self, end, pose_prop, vel, g_vec, parts_tile, pmask_tile,
                       prior_feat, mu_pair, dt_clk, mask, omit, soft):
        n = pose_prop.size(1)
        pgeom, frames, sat_vecs = self._pair_geom_mc(pose_prop, vel, g_vec, parts_tile, pmask_tile)
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
        return weights, logit, conduct, persist_p, magnitude, pgeom, frames, sat_vecs

    def _ang_directions(self, pose_prop, vel, frames, sat_vecs):
        n = pose_prop.size(1)
        e1, e2, e3 = frames
        n_i, n_j, r_i, r_j, tau_i, tau_j = sat_vecs
        dp = relative(pose_prop[..., :3])
        dv = relative(vel[..., :3])
        dp_u = dp / (1.0 + dp.norm(dim=-1, keepdim=True))
        dv_u = dv / (1.0 + dv.norm(dim=-1, keepdim=True))
        t_dv = torch.linalg.cross(r_j, dv_u, dim=-1)
        t_dp = torch.linalg.cross(r_j, dp_u, dim=-1)
        e1_j = e1[:, None].expand(-1, n, -1, -1)
        e2_j = e2[:, None].expand(-1, n, -1, -1)
        e3_j = e3[:, None].expand(-1, n, -1, -1)
        return torch.stack([tau_j, t_dv, t_dp, tau_i, e1_j, e2_j, e3_j], dim=-2)  # 7D

    def delta_messages_mc(self, end, pair_in, lin_dirs, ang_dirs, frames, g_vec, sat_vecs,
                          d_pose, d_vel, d_state, weights, d_weights, pgeom, mu_pair, time):
        n = end.size(1)
        e1, e2, e3 = frames
        n_i, n_j, r_i, r_j, tau_i, tau_j = sat_vecs
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
        e3_j = e3[:, None].expand(-1, n, -1, -1)
        g0_i, g0_j = endpoints(g_vec[:, :, 0])
        dpi = d_pose[..., :3][:, :, None].expand(-1, -1, n, -1)
        dvi = d_vel[..., :3][:, :, None].expand(-1, -1, n, -1)
        delta_lin_dirs = torch.stack([
            norm_v(dp_d), norm_v(dpi), norm_v(dv_d), norm_v(dvi),
            e1_i, e2_i, e1_j, e2_j, g0_i, g0_j, n_i, n_j, norm_v(r_j),
        ], dim=-2)  # 13D

        # Equivariant delta torque pseudovector bases
        delta_ang_dirs = torch.stack([
            torch.linalg.cross(r_j, norm_v(dv_d), dim=-1),
            torch.linalg.cross(r_j, norm_v(dvi), dim=-1),
            torch.linalg.cross(r_j, norm_v(dp_d), dim=-1),
            tau_j,
            e3_j,
            tau_i,
        ], dim=-2)  # 6D

        t_in = torch.cat([pairs(end), pgeom, diff_geom, mu_pair, time[:, None].expand(-1, n, -1, -1)], dim=-1)
        w_trans_lin = weights[..., 0] * sender_gate[:, :, None, None]
        w_trans_ang = (0.5 * (weights[..., 1] + weights[..., 2])) * sender_gate[:, :, None, None]

        c_lt = self.delta_vec_transport(t_in).unflatten(-1, (FACTORS, self.n_delta_dirs))
        c_ls = self.delta_vec_switch(pair_in).unflatten(-1, (FACTORS, self.n_dirs))
        dv_delta = (
            torch.einsum('bijf,bijfl,bijlc->bjc', w_trans_lin, c_lt, delta_lin_dirs)
            + torch.einsum('bijf,bijfl,bijlc->bjc', d_weights[..., 0], c_ls, lin_dirs)
        )

        c_at = self.delta_ang_transport(t_in).unflatten(-1, (FACTORS, self.n_delta_ang_dirs))
        c_as = self.delta_ang_switch(pair_in).unflatten(-1, (FACTORS, self.n_ang_dirs))
        dw_ang = 0.5 * (d_weights[..., 1] + d_weights[..., 2])
        domega_delta = (
            torch.einsum('bijf,bijfl,bijlc->bjc', w_trans_ang, c_at, delta_ang_dirs)
            + torch.einsum('bijf,bijfl,bijlc->bjc', dw_ang, c_as, ang_dirs)
        )
        return dv_delta, domega_delta

    def rollout_mc(self, d, facts, action, reset, ref_slot, keep=None, soft=0.0):
        x, roles = d['x'], d['role']
        b, m_br, n = action.shape[:3]
        x0 = tile(x, m_br)
        reset_flat = reset.flatten(0, 1)
        pose = x0 + reset_flat
        parts_tile = tile(d['parts'], m_br)
        pmask_tile = tile(d['part_mask'], m_br)
        roles_tile = tile(roles, m_br)
        cmd = action.flatten(0, 1)
        mask = tile(pairmask(roles), m_br)
        omit = torch.ones_like(mask) if keep is None else keep.flatten(0, 1)

        # Intervened-state layout GNN: runs on true (x0 + reset) with edge_mask = mask * omit
        # Fast path when reset==0 and keep is None (single pass + tile)
        has_reset = bool((reset_flat[..., :3].abs().max() > 1e-6).item()) if reset_flat.numel() else False
        if has_reset or (keep is not None):
            lay, g_vec = self.layout(pose, roles_tile, parts_tile, pmask_tile, edge_mask=mask * omit)
        else:
            lay0, g_vec0 = self.layout(x, roles, d['parts'], d['part_mask'])
            lay, g_vec = tile(lay0, m_br), tile(g_vec0, m_br)

        # Node context built from intervened pose & intervened layout
        m_tile = tile(facts['m'], m_br)
        u_tile = tile(facts['u'], m_br)
        mu_node = torch.cat([m_tile, u_tile], dim=-1)
        mu_pair = pairs(mu_node)
        mu_factor = mu_node[:, :, None].expand(-1, -1, FACTORS, -1)
        fact_disp = tile(d['fact'][:, -1] - d['fact'][:, 0], m_br)
        geom14 = torch.cat([parts_tile.flatten(-2), pmask_tile], dim=-1)
        body = self.node(torch.cat([
            safe_pose_input(pose, True), fact_disp, roles_tile, geom14, mu_node, lay,
        ], dim=-1))

        prior_feat = torch.tanh(tile(facts['graph'], m_br) / 3.0)
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

        # Precompute interpolated factual step velocities for same==1 Pearl exogenous abduction
        if self.use_exo:
            fact_nat = interpolate_factual_native(d['fact'], d['widths'], widths)  # (B, T_n+1, N, 9)
            v_fact_steps = (fact_nat[:, 1:] - fact_nat[:, :-1]) / widths[:, :, None, None].clamp_min(1e-3)
            same_tile = tile(d['same'], m_br)[:, None]  # (BM, 1)
            reset_norm = reset_flat[..., :3].norm(dim=-1)  # (BM, N)
            # Mark reference slot so ref_slot stays strictly open-loop!
            is_non_ref = torch.ones((b, m_br), device=x.device, dtype=x.dtype)
            is_non_ref[:, ref_slot] = 0.0
            is_non_ref = is_non_ref.flatten(0, 1)[:, None]  # (BM, 1)
            has_omit_node = ((1.0 - omit) * mask).sum(dim=-2).clamp(0.0, 1.0)  # (BM, N)
        else:
            v_fact_steps = None

        outputs = []
        for step_idx, (t_clk, dt_raw, parent_raw) in enumerate(zip(times.unbind(1), widths.unbind(1), nominal.unbind(1))):
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
            last_dw_in = None

            for _ in range(self.hops):
                prop_in = safe_pose_input(proposed, True)
                end = self.endpoint(torch.cat([
                    state.flatten(-2), message.flatten(-2), prop_in, v_curr, free_acc, mu_node, lay,
                ], dim=-1))
                weights, logit_h, conduct_h, persist_h, strength_h, pgeom, frames, sat_vecs = self.interaction_mc(
                    end, proposed, v_curr, g_vec, parts_tile, pmask_tile,
                    prior_feat, mu_pair, time[:, 0], mask, omit, soft)

                base_msg = torch.einsum('bijfg,bifh->bjgh', weights, self.message(state))
                n_i, n_j, r_i, r_j, tau_i, tau_j = sat_vecs
                base_dirs = self._base_directions(pose, proposed, v_curr, frames, g_vec)
                r_j_unit = r_j / (1.0 + r_j.norm(dim=-1, keepdim=True))
                lin_dirs = torch.cat([base_dirs, n_i.unsqueeze(-2), n_j.unsqueeze(-2), r_j_unit.unsqueeze(-2)], dim=-2)
                ang_dirs = self._ang_directions(proposed, v_curr, frames, sat_vecs)

                dv_vec, pair_in = self.vector_message(end, lin_dirs, pgeom, mu_pair, time, weights)
                # Equivariant angular velocity impulse from contact torque
                ang_coef = self.ang_vector(pair_in).unflatten(-1, (FACTORS, self.n_ang_dirs))
                w_ang = 0.5 * (weights[..., 1] + weights[..., 2])
                domega = torch.einsum('bijf,bijfl,bijlc->bjc', w_ang, ang_coef, ang_dirs)

                if self.use_delta_prop or self.use_exo:
                    w_b = weights.unflatten(0, (b, m_br))
                    w_ref = w_b[:, ref_slot:ref_slot + 1].expand_as(w_b).flatten(0, 1)
                    d_weights = (weights - w_ref) * omit[..., None, None]
                    last_dw_in = d_weights.abs().sum(dim=(-1, -2)).sum(dim=-2)  # (BM, N) incoming edge-change L1

                if self.use_delta_prop:
                    st_b = state.unflatten(0, (b, m_br))
                    st_ref = st_b[:, ref_slot:ref_slot + 1].expand_as(st_b).flatten(0, 1)
                    d_state = state - st_ref

                    po_b = proposed.unflatten(0, (b, m_br))
                    d_pose = (po_b - po_b[:, ref_slot:ref_slot + 1]).flatten(0, 1)

                    ve_b = v_curr.unflatten(0, (b, m_br))
                    d_vel = (ve_b - ve_b[:, ref_slot:ref_slot + 1]).flatten(0, 1)

                    delta_msg = (
                        torch.einsum('bijfg,bifh->bjgh', weights, self.msg_delta(d_state))
                        + torch.einsum('bijfg,bifh->bjgh', d_weights, self.msg_switch(st_ref))
                    )
                    delta_dv, delta_domega = self.delta_messages_mc(
                        end, pair_in, lin_dirs, ang_dirs, frames, g_vec, sat_vecs,
                        d_pose, d_vel, d_state, weights, d_weights, pgeom, mu_pair, time,
                    )
                    message = base_msg + delta_msg
                    dv_vec = dv_vec + delta_dv
                    domega = domega + delta_domega
                else:
                    message = base_msg

                rec_in = torch.cat([message, mu_factor, node_time], dim=-1).flatten(0, 2)
                h_cand = self.recurrent(rec_in, state.flatten(0, 2)).unflatten(0, state.shape[:3])
                state = self.state_norm((1.0 - alpha_hop) * state + alpha_hop * h_cand)

                dv_resp = torch.cat([
                    f(torch.cat([state[:, :, i_f], message[:, :, i_f], mu_node, time], dim=-1))
                    for i_f, f in enumerate(self.response)
                ], dim=-1)
                # Rigid-body equivariant orientation kinematics: d(e1)/dt += domega x e1, d(e2)/dt += domega x e2
                e1_f, e2_f, _ = frames
                domega_clamped = domega.clamp(-8.0, 8.0)
                dv_e1 = torch.linalg.cross(domega_clamped, e1_f, dim=-1)
                dv_e2 = torch.linalg.cross(domega_clamped, e2_f, dim=-1)
                dv_hop = torch.cat([
                    dv_vec + dv_resp[..., :3],
                    dv_e1 + dv_resp[..., 3:6],
                    dv_e2 + dv_resp[..., 6:9],
                ], dim=-1)
                v_curr = (v_curr + dv_hop / self.hops).clamp(-50.0, 50.0)
                proposed = pose + dt * v_curr

                hop_logits.append(logit_h)
                hop_conducts.append(conduct_h)
                hop_persists.append(persist_h)
                hop_strengths.append(strength_h)

            # Pearl divergence-gated step-wise exogenous abduction (strictly for same==1 & non-ref branches!)
            if self.use_exo:
                v_ref_b = v_curr.unflatten(0, (b, m_br))[:, ref_slot:ref_slot + 1].detach()
                p_ref_b = proposed.unflatten(0, (b, m_br))[:, ref_slot:ref_slot + 1].detach()
                v_ref_t = v_ref_b.expand(b, m_br, n, 9).flatten(0, 1)
                p_ref_t = p_ref_b.expand(b, m_br, n, 9).flatten(0, 1)
                v_fact_t = tile(v_fact_steps[:, step_idx], m_br)
                e_exo = (v_fact_t - v_ref_t).clamp(-15.0, 15.0)

                dp_div = (proposed[..., :3] - p_ref_t[..., :3]).norm(dim=-1)
                dv_div = (v_curr[..., :3] - v_ref_t[..., :3]).norm(dim=-1)
                dw_div = last_dw_in if last_dw_in is not None else torch.zeros_like(dp_div)
                div_feat = torch.stack([dp_div, dv_div, dw_div, reset_norm, has_omit_node], dim=-1)
                gate_logits = self.exo_gate(torch.cat([div_feat, mu_node], dim=-1))
                decay_rate = F.softplus(self.exo_log_decay)
                div_mag = dp_div + 0.5 * dv_div + 0.5 * dw_div + 2.0 * reset_norm + 3.0 * has_omit_node
                env = torch.exp(-decay_rate * div_mag)[..., None] * (same_tile * is_non_ref)[..., None]
                g_pos = gate_logits[..., 0:1].sigmoid() * env
                g_rot = gate_logits[..., 1:2].sigmoid() * env
                exo_corr = torch.cat([g_pos * e_exo[..., :3], g_rot * e_exo[..., 3:]], dim=-1)
                v_curr = (v_curr + exo_corr).clamp(-50.0, 50.0)
                proposed = pose + dt * v_curr

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
        out = self.rollout_mc(centered, facts, action_all, reset_all, ref_slot, keep, soft)
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
