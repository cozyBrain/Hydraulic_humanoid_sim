"""Hydraulic system model of the v9 humanoid (JAX, vectorisable, differentiable).

Per actuator (29): 4-way servo valve (spool lag + orifice flows), two compressible chambers (bulk modulus,
hose dead volume, stroke-dependent volume), piston/vane cross-leakage, A-B proportional bypass valve,
seal friction, end stops.  Cylinders act through exact kinematics (kinematics.py); vanes act directly.
Supply: BLDC + 10 cc/rev internal gear pump (torque & power limited, speed loop on pressure error),
2 L piston accumulator (polytropic gas), relief valve, oil temperature (throttling + pump losses vs cooler),
battery energy / current.

Units SI.  Pressures gauge [Pa].  Joint order (29): waist_yaw, waist_pitch, waist_roll,
L: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll, R: same."""
import numpy as np, jax, jax.numpy as jnp
from kinematics import knee_len_arm, ach_len_jac, waist_len_jac, elbow_len_arm, KNEE_LMIN, KNEE_LMAX, ACH_L0, WAIST_L0, ELBOW_LMIN, ELBOW_LMAX

ARMJ = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'forearm_roll', 'wrist_pitch', 'wrist_yaw')
JOINTS = (['waist_yaw', 'waist_pitch', 'waist_roll'] + [f'{j}_{s}' for s in 'LR' for j in ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle_pitch', 'ankle_roll')]
          + [f'{j}_{s}' for s in 'LR' for j in ARMJ])
ACTS = (['waist_yaw_vane', 'waist_cyl_L', 'waist_cyl_R'] + [f'{a}_{s}' for s in 'LR' for a in ('hip_yaw_vane', 'hip_roll_vane', 'hip_pitch_vane', 'knee_cyl', 'ach_cyl_0', 'ach_cyl_1')]
        + [f'{a}_{s}' for s in 'LR' for a in ('sh_pitch_vane', 'sh_roll_vane', 'sh_yaw_vane', 'triceps_cyl', 'forearm_roll_vane', 'wrist_pitch_vane', 'wrist_yaw_vane')])
NA = len(JOINTS)          # 29 actuated joints, one actuator each (the 2-cylinder ankles / waist share their 2 DOF)
ARM0 = 15
BORE_T, ROD_T = 0.0246, 0.0144             # triceps
def area(d): return np.pi * d * d / 4
BORE_Q, ROD_Q = 0.02706, 0.01584           # knee + waist cylinders
BORE_A, ROD_A = 0.01919, 0.01123           # Achilles
DEG = np.pi / 180

def _p(**kw): return {k: np.array(v, dtype=np.float32) for k, v in kw.items()}
def default_params():
    is_cyl = np.zeros(NA); AA = np.zeros(NA); AB = np.zeros(NA); xmin = np.zeros(NA); stroke = np.zeros(NA); Qr = np.zeros(NA)
    def vane(i, D, lo, hi, q):  AA[i] = AB[i] = D; xmin[i] = lo * DEG; stroke[i] = (hi - lo) * DEG; Qr[i] = q
    def cyl(i, b, r, lmin, st, q): is_cyl[i] = 1; AA[i] = area(b); AB[i] = area(b) - area(r); xmin[i] = lmin; stroke[i] = st; Qr[i] = q
    vane(0, 8e-6, -40, 40, 8)
    for i in (1, 2): cyl(i, BORE_Q, ROD_Q, WAIST_L0 - 0.035, 0.070, 10)
    for k, s in enumerate('LR'):
        a = 3 + 6 * k
        vane(a, 6e-6, -40, 40, 8); vane(a + 1, 1.2e-5, -40, 40, 15); vane(a + 2, 1.6e-5, -30, 120, 20)
        cyl(a + 3, BORE_Q, ROD_Q, KNEE_LMIN - 0.004, KNEE_LMAX - KNEE_LMIN + 0.008, 20)
        cyl(a + 4, BORE_A, ROD_A, ACH_L0 - 0.040, 0.080, 10); cyl(a + 5, BORE_A, ROD_A, ACH_L0 - 0.040, 0.080, 10)
    for k, s in enumerate('LR'):
        a = ARM0 + 7 * k
        vane(a, 1.2e-5, -60, 170, 15); vane(a + 1, 1.0e-5, -15, 110, 12); vane(a + 2, 5e-6, -90, 90, 8)
        cyl(a + 3, BORE_T, ROD_T, ELBOW_LMIN - 0.004, ELBOW_LMAX - ELBOW_LMIN + 0.008, 12)
        vane(a + 4, 3e-6, -90, 90, 6); vane(a + 5, 3e-6, -70, 70, 6); vane(a + 6, 2e-6, -40, 40, 5)
    Lmin_per_l = 1 / 60000
    # real valves (BOM option B, Moog DDV cartridges): rated flow @70 bar and catalogue maximum flow
    #   Qr class 12-20 -> E242-200A 18 L/min (max 18) | 10 -> E242-205A 11 (max 18) | 6-8 -> E245-105 8.5 (max 10) | 5 -> E245-104 5.5 (max 10)
    VALVE = {20: (18, 18), 15: (18, 18), 12: (18, 18), 10: (11, 18), 8: (8.5, 10), 6: (8.5, 10), 5: (5.5, 10)}
    Qmax = np.array([VALVE[int(q)][1] for q in Qr]); Qr = np.array([VALVE[int(q)][0] for q in Qr], float)
    return dict(
        Qmax=(Qmax * Lmin_per_l).astype(np.float32),                      # valve flow ceiling (port/flow-force limit), m3/s
        is_cyl=is_cyl.astype(np.float32), AA=AA.astype(np.float32), AB=AB.astype(np.float32), xmin=xmin.astype(np.float32), stroke=stroke.astype(np.float32),
        Kv=(Qr * Lmin_per_l / np.sqrt(35e5)).astype(np.float32),          # orifice coeff per land, rated flow at 35 bar/land
        V0=np.where(is_cyl > 0, 1.6e-5, 1.0e-5).astype(np.float32),       # dead volume per chamber (ports + hose)
        Kb=(Qr * Lmin_per_l / np.sqrt(35e5) * 0.6).astype(np.float32),    # A-B bypass orifice at full open
        Cl=np.full(NA, 1.5e-13, np.float32),                               # cross-port leakage m3/s/Pa
        Fc=np.where(is_cyl > 0, 40.0, 1.5).astype(np.float32), Fv=np.where(is_cyl > 0, 300.0, 0.5).astype(np.float32),
        **_p(beta=7.0e8, tau_v=0.003, Pt=0.5e5, Pmax=3.5e7,
             Dp=15.8e-6 / (2 * np.pi), eta_m=0.9, slip=0.08 / 2.1e7, Jm=1.52e-2, wmax=4000 * 2 * np.pi / 60, Tmax=58.5, Pel_max=24000.0, Ksp=4.0,   # EMRAX 188 LV + EMSISO L30B (450 A x 0.13 Nm/A), Eckerle EIPS2-016 (15.8 cc, max 4000 rpm); Pel 24 kW = L30B 58.5 Nm x 4000 rpm
             Ps_set=2.07e7, Kw=0.002, Kps=40.0 / 1e5, Prel=2.3e7, Krel=1.0e-10,
             Vacc=2.0e-3, p0=1.4e7, npoly=1.3, Vsoft=2.0e-5,
             Cth=9000.0, UA=60.0, Tamb=25.0, Vbat=88.0, Rbat=0.04, Paux=150.0, eta_e=0.92))

def init_state(P, q, batch_shape=()):
    """hydraulic state at rest, chambers balanced at mid supply pressure"""
    z = jnp.zeros(batch_shape + (NA,))
    Voil0 = P['Vacc'] * (1 - (P['p0'] / P['Ps_set']) ** (1 / P['npoly']))
    return dict(pA=z + 0.4 * P['Ps_set'], pB=z + 0.4 * P['Ps_set'] * P['AA'] / P['AB'], xv=z,
                Voil=jnp.zeros(batch_shape) + Voil0, w=jnp.zeros(batch_shape) + 400.0, T=jnp.zeros(batch_shape) + 35.0, E=jnp.zeros(batch_shape))

def actuator_jac(q):
    """J (15 act x 15 joints) with d(actuator displacement)/dq, and actuator displacement x (m or rad)"""
    J = jnp.zeros((NA, NA)); x = jnp.zeros(NA)
    J = J.at[0, 0].set(1.0); x = x.at[0].set(q[0])
    Lw, Jw = waist_len_jac(q[1], q[2]); J = J.at[1:3, 1:3].set(Jw); x = x.at[1:3].set(Lw)
    for k, s in enumerate('LR'):
        a = 3 + 6 * k
        for i in range(3): J = J.at[a + i, a + i].set(1.0); x = x.at[a + i].set(q[a + i])
        Lk, ak = knee_len_arm(q[a + 3]); J = J.at[a + 3, a + 3].set(ak); x = x.at[a + 3].set(Lk)
        La, Ja = ach_len_jac(q[a + 4], q[a + 5], s); J = J.at[a + 4:a + 6, a + 4:a + 6].set(Ja); x = x.at[a + 4:a + 6].set(La)
    for k in range(2):
        a = ARM0 + 7 * k
        for i in (0, 1, 2, 4, 5, 6): J = J.at[a + i, a + i].set(1.0); x = x.at[a + i].set(q[a + i])
        Le, ae = elbow_len_arm(q[a + 3]); J = J.at[a + 3, a + 3].set(ae); x = x.at[a + 3].set(Le)
    return x, J

def sflow(dp, eps=2e5):   # signed sqrt with laminar region below ~2 bar (keeps the ODE smooth)
    return dp / jnp.sqrt(jnp.abs(dp) + eps)

def step(P, h, q, qd, u_spool, u_bypass, dt, motor_on=1.0):
    """advance hydraulics by dt given joint q, qd.  returns new state, joint torques (29), diagnostics"""
    x_abs, J = actuator_jac(q)
    x = x_abs - P['xmin']; v = J @ qd                                          # piston/vane position and velocity
    # supply node: accumulator gas law (soft collapse when empty)
    Vg = jnp.maximum(P['Vacc'] - h['Voil'], 0.05 * P['Vacc'])
    Ps = jnp.where(h['Voil'] >= 0, P['p0'] * (P['Vacc'] / Vg) ** P['npoly'], P['p0'] * jnp.maximum(1 + h['Voil'] / P['Vsoft'], 0.005))
    Ps = Ps * motor_on + (1 - motor_on) * jnp.minimum(Ps, P['p0'])           # (power off: accumulator isolated by check/dump valve)
    Pt = P['Pt']; pA, pB = h['pA'], h['pB']
    # servo valve: spool lag, critical-centre 4-way orifice flows
    xv = h['xv'] + (jnp.clip(u_spool, -1, 1) * motor_on - h['xv']) * jnp.minimum(dt / P['tau_v'], 1.0)
    xp, xn = jnp.maximum(xv, 0), jnp.maximum(-xv, 0)
    visc = jnp.exp(0.03 * (h['T'] - 40.0))                                      # warmer oil leaks more
    QA = P['Kv'] * (xp * sflow(Ps - pA) - xn * sflow(pA - Pt))
    QB = P['Kv'] * (xn * sflow(Ps - pB) - xp * sflow(pB - Pt))
    QA = jnp.clip(QA, -P['Qmax'], P['Qmax']); QB = jnp.clip(QB, -P['Qmax'], P['Qmax'])     # real valves saturate (E242 18, E245 10 L/min)
    Qnull = 2e-14 * visc * ((Ps - pA) + (Ps - pB))                              # spool null leakage (supply -> chambers ~ tank)
    Qleak = P['Cl'] * visc * (pA - pB)
    byp = jnp.clip(u_bypass + (1 - motor_on), 0, 1)
    Qby = P['Kb'] * byp * sflow(pA - pB)
    # chamber pressure dynamics
    VA = P['V0'] + P['AA'] * jnp.clip(x, 0, P['stroke']); VB = P['V0'] + P['AB'] * jnp.clip(P['stroke'] - x, 0, P['stroke'])
    dpA = P['beta'] / VA * (QA - P['AA'] * v - Qleak - Qby)
    dpB = P['beta'] / VB * (QB + P['AB'] * v + Qleak + Qby)
    pA_n = jnp.clip(pA + dpA * dt, 0.0, P['Pmax']); pB_n = jnp.clip(pB + dpB * dt, 0.0, P['Pmax'])
    # actuator force / torque (pressures from start of step: semi-implicit with MuJoCo's velocity update)
    F = pA * P['AA'] - pB * P['AB'] - P['Fc'] * jnp.tanh(v / jnp.where(P['is_cyl'] > 0, 0.005, 0.02)) - P['Fv'] * v
    kstop = jnp.where(P['is_cyl'] > 0, 4e6, 2e4)
    F = F + kstop * (jnp.maximum(-x, 0) - jnp.maximum(x - P['stroke'], 0)) - 0.02 * kstop * v * ((x < 0) | (x > P['stroke']))
    tau = J.T @ F
    # supply: pump + motor + accumulator + relief
    Qdraw = jnp.sum(P['Kv'] * (xp * jnp.maximum(sflow(Ps - pA), 0) + xn * jnp.maximum(sflow(Ps - pB), 0))) + jnp.sum(Qnull)
    w = h['w']
    w_cmd = jnp.clip((P['Ps_set'] - Ps) / 10e5 + 0.1, 0, 1) * P['wmax'] * motor_on     # 10 bar proportional band, 10% idle flow
    Tm = jnp.clip(P['Ksp'] * (w_cmd - w), -P['Tmax'], P['Tmax'])
    Tm = jnp.clip(Tm, -P['Pel_max'] / jnp.maximum(w, 50.0), P['Pel_max'] / jnp.maximum(w, 50.0)) * motor_on
    Tload = Ps * P['Dp'] / P['eta_m'] + 0.004 * w
    w_n = jnp.clip(w + (Tm - Tload) / P['Jm'] * dt, 0.0, 1.05 * P['wmax'])
    Qpump = P['Dp'] * w * (1 - P['slip'] * Ps * visc)
    Qrel = P['Krel'] * jnp.maximum(Ps - P['Prel'], 0)
    Voil_n = jnp.minimum(h['Voil'] + (Qpump - Qdraw - Qrel) * dt, 0.92 * P['Vacc'])
    # energy & heat
    Pmech_pump = Tm * w; Pel = jnp.maximum(Pmech_pump, 0) / P['eta_e'] + P['Paux']
    Pdeliv = jnp.sum(F * v)
    Ploss = jnp.maximum(Pel - P['Paux'] - Pdeliv, 0.0)
    T_n = h['T'] + (Ploss - P['UA'] * (h['T'] - P['Tamb'])) / P['Cth'] * dt
    E_n = h['E'] + Pel * dt
    Ibat = Pel / P['Vbat']
    hn = dict(pA=pA_n, pB=pB_n, xv=xv, Voil=Voil_n, w=w_n, T=T_n, E=E_n)
    diag = dict(Ps=Ps, F=F, x=x, v=v, Qpump=Qpump, Qdraw=Qdraw, Pel=Pel, Ibat=Ibat, Tm=Tm)
    return hn, tau, diag

# ---------------------------------------------------------------- on-robot low-level controller (1 kHz firmware)
def force_from_torque(J, tau_des, P):
    """distribute desired joint torques to actuator forces: tau = J^T F  ->  F = J^-T tau (block diagonal, well conditioned)"""
    return jnp.linalg.solve(J.T + 1e-6 * jnp.eye(NA), tau_des)

def servo_command(P, h, q, qd, tau_des, J, diag_v, kf=None, kff=None):
    """pressure-based force control: spool = feed-forward (flow for the current velocity) + force-error feedback.
    The force feedback uses the measured chamber pressures (the same sensors the real robot has)."""
    F_des = force_from_torque(J, tau_des, P)
    F_meas = h['pA'] * P['AA'] - h['pB'] * P['AB']
    Fmax = P['Ps_set'] * P['AA']
    if kf is None:   # per-actuator force-loop gain for a ~120 rad/s crossover (valve lag 3 ms, 1 kHz loop)
        Vmid = P['V0'] + 0.5 * P['AA'] * P['stroke']
        plant = P['Kv'] * jnp.sqrt(0.5 * P['Ps_set']) * P['beta'] * P['AA'] / Vmid * 2.0
        kf = 250.0 * Fmax / plant
    ff = (P['AA'] * diag_v) / (P['Kv'] * jnp.sqrt(0.5 * P['Ps_set']) + 1e-12)
    kff = 0.7 if kff is None else kff      # velocity compensation (cancels the A*v flow term; <1 keeps it stable)
    return jnp.clip(kf * (F_des - F_meas) / Fmax + kff * ff, -1, 1)

def fluid_armature(P, q, L_hose=0.6, d_hose=0.00635, rho=870.0):
    """reflected inertia of the oil columns in the A and B hoses (fluid inertance) -> joint armature [kg m^2]"""
    Ah = area(d_hose); m_col = rho * L_hose * Ah
    x, J = actuator_jac(jnp.array(q))
    meff = m_col * ((P['AA'] / Ah) ** 2 + (P['AB'] / Ah) ** 2)        # per actuator, at the piston / vane
    return np.array(jnp.einsum('i,ij,ij->j', meff, J, J))

def forces(P, h, q, qd):
    """actuator forces from the CURRENT chamber pressures -> joint torques.  Used before the rigid-body step;
    hy.step() is then called with the post-step q, qd (symplectic coupling: stable at 1 ms for the stiff oil columns)."""
    x_abs, J = actuator_jac(q); x = x_abs - P['xmin']; v = J @ qd
    F = h['pA'] * P['AA'] - h['pB'] * P['AB'] - P['Fc'] * jnp.tanh(v / jnp.where(P['is_cyl'] > 0, 0.005, 0.02)) - P['Fv'] * v
    kstop = jnp.where(P['is_cyl'] > 0, 4e6, 2e4)
    F = F + kstop * (jnp.maximum(-x, 0) - jnp.maximum(x - P['stroke'], 0)) - 0.02 * kstop * v * ((x < 0) | (x > P['stroke']))
    return J.T @ F, F, J, v

# ======================================================================================================
# FAST (reduced-order) actuator model for stage-1 training at 4 ms steps.
# Same firmware input (desired joint torque).  Keeps the physics that shapes behaviour:
#   * force limits from supply pressure x piston/annulus area (or vane displacement), mapped through the exact kinematics
#   * valve flow limit -> force collapses as actuator speed approaches Kv*sqrt(Ps/2)/A in the driving direction
#   * force response lag (pressure build-up through the valve) ~ 6 ms
#   * pump / accumulator supply: flow demand drains the accumulator, BLDC speed & power limits, pressure sag
#   * friction, end stops
# Dropped: chamber compressibility oscillations, spool dynamics, cross-leakage (stage 2 fine-tunes on the full model).
# ======================================================================================================
def fast_init(P, batch_shape=()):
    z = jnp.zeros(batch_shape + (NA,))
    Voil0 = P['Vacc'] * (1 - (P['p0'] / P['Ps_set']) ** (1 / P['npoly']))
    return dict(F=z, Fdes=z, Voil=jnp.zeros(batch_shape) + Voil0, w=jnp.zeros(batch_shape) + 400.0, T=jnp.zeros(batch_shape) + 35.0, E=jnp.zeros(batch_shape))

def fast_forces(P, h, q, qd):
    x_abs, J = actuator_jac(q); x = x_abs - P['xmin']; v = J @ qd
    F = h['F'] - P['Fc'] * jnp.tanh(v / jnp.where(P['is_cyl'] > 0, 0.005, 0.02)) - P['Fv'] * v
    # (no oil-column end stops here: at 4 ms they would be numerically stiff; MuJoCo's joint limits hold the range)
    return J.T @ F, F, J, v

def fast_step(P, h, q, qd, tau_des, dt, motor_on=1.0, tau_f=0.006):
    _, J = actuator_jac(q); v = J @ qd
    Vg = jnp.maximum(P['Vacc'] - h['Voil'], 0.05 * P['Vacc'])
    Ps = jnp.where(h['Voil'] >= 0, P['p0'] * (P['Vacc'] / Vg) ** P['npoly'], P['p0'] * jnp.maximum(1 + h['Voil'] / P['Vsoft'], 0.005)) * motor_on
    Fdes = jnp.linalg.solve(J.T + 1e-6 * jnp.eye(NA), tau_des)
    Fpos = Ps * P['AA'] - P['Pt'] * P['AB']; Fneg = -(Ps * P['AB'] - P['Pt'] * P['AA'])
    Qv = jnp.minimum(P['Kv'] * jnp.sqrt(0.5 * Ps + 1e3), P['Qmax'])               # valve flow incl. catalogue ceiling
    vmax_ext = Qv / P['AA']; vmax_ret = Qv / P['AB']
    sc_pos = jnp.where(v > 0, jnp.clip(1 - (v / vmax_ext) ** 2, 0, 1), 1.0)      # pushing while extending: flow limited
    sc_neg = jnp.where(v < 0, jnp.clip(1 - (v / vmax_ret) ** 2, 0, 1), 1.0)
    Ft = jnp.clip(Fdes, Fneg * sc_neg, Fpos * sc_pos)
    F = h['F'] + (Ft - h['F']) * jnp.minimum(dt / tau_f, 1.0)
    # supply
    # supply flow is only drawn when the actuator DRIVES the motion (F*v > 0). When the load back-drives it
    # (braking, falling, gravity lowering a limb) the valve meters the compressed chamber to tank / back towards
    # the supply and the expanding chamber fills from the return side -> no accumulator drain (as in the full model).
    Qdraw = jnp.sum(jnp.where(h['F'] * v > 0, jnp.where(v > 0, P['AA'], P['AB']) * jnp.abs(v), 0.0)) + 6e-6
    w = h['w']
    w_cmd = jnp.clip((P['Ps_set'] - Ps) / 10e5 + 0.1, 0, 1) * P['wmax'] * motor_on
    Tm = jnp.clip(P['Ksp'] * (w_cmd - w), -P['Tmax'], P['Tmax'])
    Tm = jnp.clip(Tm, -P['Pel_max'] / jnp.maximum(w, 50.0), P['Pel_max'] / jnp.maximum(w, 50.0)) * motor_on
    w_n = jnp.clip(w + (Tm - (Ps * P['Dp'] / P['eta_m'] + 0.004 * w)) / P['Jm'] * dt, 0.0, 1.05 * P['wmax'])
    Qpump = P['Dp'] * w * (1 - P['slip'] * Ps)
    Qrel = P['Krel'] * jnp.maximum(Ps - P['Prel'], 0)
    Voil_n = jnp.minimum(h['Voil'] + (Qpump - Qdraw - Qrel) * dt, 0.92 * P['Vacc'])
    Pel = jnp.maximum(Tm * w, 0) / P['eta_e'] + P['Paux']
    Ploss = jnp.maximum(Pel - P['Paux'] - jnp.sum(F * v), 0.0)
    T_n = h['T'] + (Ploss - P['UA'] * (h['T'] - P['Tamb'])) / P['Cth'] * dt
    hn = dict(F=F, Fdes=Fdes, Voil=Voil_n, w=w_n, T=T_n, E=h['E'] + Pel * dt)
    return hn, dict(Ps=Ps, Pel=Pel, Ibat=Pel / P['Vbat'])
