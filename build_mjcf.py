"""Build the MuJoCo model of the v9 hydraulic humanoid.
- rigid bodies with mass/inertia lumped from the CAD (extract_bodies.py)
- joints at the real axes; the knee 4-bar and the Achilles / waist cylinders are NOT separate MuJoCo bodies:
  their exact kinematics (length + moment arm vs joint angle) live in kinematics.py and the hydraulic
  cylinder force is mapped to joint torque through the exact Jacobian (their few hundred grams are lumped).
- writes: model/hydro_humanoid.xml (training, primitive collision only) and
          model/hydro_humanoid_viz.xml (+ decimated CAD meshes and moving linkage parts as mocap bodies)"""
import json, os, numpy as np, trimesh, fast_simplification
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = json.load(open(f'{HERE}/out/bodies_raw.json'))

from legs import U
# ---- real-part masses (BOM 2026-11, baseline): CAD masses -> purchasable parts, as point masses at the part locations.
# (three.js design coords; MuJoCo body frame = (z, x, y) * 0.15).  Hoses are left at the CAD value (2.5 kg conservative).
REAL_PARTS = [
    ('torso', 9.438, (0, 4.25, 0.3), 'servo valves: Moog E242 x18 + E245 x14 (10.3 kg) + AM manifold, sim had 3.9 kg'),
    ('torso', 3.35, (0, 4.25, 0.3), '67 pressure sensors (TE M3200 class) in the manifold'),
    ('torso', 0.49, (0, 4.25, 0.3), 'relief / unload / check / bleed cartridges'),
    ('torso', 1.4, (1.05, 2.84, 0.45), 'HYDAC SBO 1 L accumulator 3.6 kg (sim 2.2)'),
    ('torso', 1.4, (-1.05, 2.84, 0.35), 'HYDAC SBO 1 L accumulator 3.6 kg (sim 2.2)'),
    ('torso', 5.734, (0, 3.33, -1.62), 'battery: 216 x Molicel P45B 15.1 kg + pack hardware 2.5 kg (sim 11.9)'),
    ('torso', 2.287, (0.98, 4.79, -0.66), 'EMSISO L30B inverter 2.6 kg (sim 0.3)'),
    ('torso', 2.336, (1.15, 4.0, 0.4), 'Eckerle EIPS2-016 3.6 kg + coupling 0.4 kg (sim 1.7)'),
    ('torso', 0.914, (0, 2.0, 1.0), 'Setrab ProLine cooler 1.04 kg + fan'),
    ('torso', 0.674, (-1.45, 3.5, 0.05), 'pressure + return filters'),
    ('torso', -0.409, (-1.15, 4.5, -0.1), 'pressurised piston reservoir 1.5 kg (sim 1.9)'),
    ('torso', 0.46, (0, 4.25, 0.3), 'oil outside the hoses'),
    ('torso', 3.2, (0, 3.0, -1.3), '2 x Mean Well SD-500H-24 DC-DC + BMS + contactor/fuses'),
    ('torso', 3.0, None, 'wiring harness (spread over the torso)'),
    ('foot_L', 0.508, None, 'ATI Mini85 F/T 0.635 kg (sim 0.127)'),
    ('foot_R', 0.508, None, 'ATI Mini85 F/T 0.635 kg (sim 0.127)'),
]
def _add_point(body, dm, p3):
    B = RAW[body]; M = B['mass']; c = np.array(B['com']); I = np.array(B['inertia'])
    p = c if p3 is None else np.array([p3[2], p3[0], p3[1]]) * U
    m2 = M + dm; c2 = (M * c + dm * p) / m2
    def pa(m, r): r = r - c2; return m * (r @ r * np.eye(3) - np.outer(r, r))
    B['mass'] = m2; B['com'] = c2.tolist(); B['inertia'] = (I + pa(M, c) + pa(dm, p)).tolist()
_m0 = sum(b['mass'] for b in RAW.values())
for body, dm, p3, _ in REAL_PARTS: _add_point(body, dm, p3)
print(f'real-part masses: {_m0:.2f} -> {sum(b["mass"] for b in RAW.values()):.2f} kg')
OUTD = f'{HERE}/model'; os.makedirs(OUTD + '/meshes', exist_ok=True)
from legs import LY, ZG, U, DT                    # design unit, ground offset (sole bottom at three y = LY(-4.895))
def P(x3, y3, z3): return np.array([z3 * U, x3 * U, y3 * U + ZG])
def sh(v): return np.array(v) + np.array([0, 0, ZG])        # raw extraction coords -> grounded
KN = json.load(open(f'{HERE}/../knee_sol3_21.json'))

LUMP = {'uarm': ['tribarrel'], 'farm': ['trirod'], 'thigh': ['rocker', 'coupler', 'quadbarrel', 'quadrod'], 'shin': ['achbarrel_{s}0', 'achbarrel_{s}1'], 'foot': ['achrod_{s}0', 'achrod_{s}1']}
def combine(names):
    m = sum(RAW[n]['mass'] for n in names); c = sum(RAW[n]['mass'] * sh(RAW[n]['com']) for n in names) / m
    I = np.zeros((3, 3))
    for n in names:
        d = sh(RAW[n]['com']) - c; I += np.array(RAW[n]['inertia']) + RAW[n]['mass'] * (d @ d * np.eye(3) - np.outer(d, d))
    w, _ = np.linalg.eigh(I); I = I + np.eye(3) * max(0, 1e-5 - w.min())
    return m, c, I

# ---------------------------------------------------------------- tree
lx = 0.85
J = dict(
    pelvis=P(0, 0.3, -0.2), carrier=P(0, 0.55, -0.2), spider=P(0, 1.1, -0.2), torso=P(0, 1.1, -0.2),
    **{f'hipyaw_{s}': P(sg * lx, 0.02, 0.05) for s, sg in (('L', 1), ('R', -1))},
    **{f'hiproll_{s}': P(sg * lx, -0.36, 0.05) for s, sg in (('L', 1), ('R', -1))},
    **{f'thigh_{s}': P(sg * lx, -0.94, 0.05) for s, sg in (('L', 1), ('R', -1))},
    **{f'shin_{s}': P(sg * lx, LY(-2.65), 0.05) for s, sg in (('L', 1), ('R', -1))},
    **{f'ankspider_{s}': P(sg * lx, LY(-4.5), 0.05) for s, sg in (('L', 1), ('R', -1))},
    **{f'foot_{s}': P(sg * lx, LY(-4.5), 0.05) for s, sg in (('L', 1), ('R', -1))},
    **{f'shpitch_{s}': P(sg * 2.42, 5.35, -0.2) for s, sg in (('L', 1), ('R', -1))},
    **{f'shroll_{s}': P(sg * 2.98, 5.35, -0.2) for s, sg in (('L', 1), ('R', -1))},
    **{f'uarm_{s}': P(sg * 2.98, 4.94, -0.2) for s, sg in (('L', 1), ('R', -1))},
    **{f'farm_{s}': P(sg * 2.98, 3.15, -0.2) for s, sg in (('L', 1), ('R', -1))},
    **{f'farm2_{s}': P(sg * 2.98, 2.62, -0.2) for s, sg in (('L', 1), ('R', -1))},
    **{f'wrist_{s}': P(sg * 2.98, 1.5, -0.2) for s, sg in (('L', 1), ('R', -1))},
    **{f'flange_{s}': P(sg * 2.98, 1.14, -0.2) for s, sg in (('L', 1), ('R', -1))})
#   joint name, axis (mj), range (deg)   -- signs: hip_pitch + = flexion (thigh forward), knee + = flexion,
#   ankle_pitch + = toes down (plantarflexion), ankle_roll + = sole tilts toward +x-axis rotation
JNT = {'carrier': ('waist_yaw', (0, 0, 1), (-40, 40)), 'spider': ('waist_pitch', (0, 1, 0), (-25, 25)), 'torso': ('waist_roll', (1, 0, 0), (-20, 20))}
for s in 'LR':
    JNT[f'hipyaw_{s}'] = (f'hip_yaw_{s}', (0, 0, 1), (-40, 40))
    JNT[f'hiproll_{s}'] = (f'hip_roll_{s}', (1, 0, 0), (-25, 40) if s == 'L' else (-40, 25))
    JNT[f'thigh_{s}'] = (f'hip_pitch_{s}', (0, -1, 0), (-30, 120))
    JNT[f'shin_{s}'] = (f'knee_{s}', (0, 1, 0), (0, 140))
    JNT[f'ankspider_{s}'] = (f'ankle_pitch_{s}', (0, 1, 0), (-35, 35))
    JNT[f'foot_{s}'] = (f'ankle_roll_{s}', (1, 0, 0), (-22, 22))
    sg = 1 if s == 'L' else -1          # arm: + = forward raise / abduction / flexion, mirrored for the right side
    JNT[f'shpitch_{s}'] = (f'shoulder_pitch_{s}', (0, -1, 0), (-60, 170))
    JNT[f'shroll_{s}'] = (f'shoulder_roll_{s}', (sg, 0, 0), (-15, 110))
    JNT[f'uarm_{s}'] = (f'shoulder_yaw_{s}', (0, 0, sg), (-90, 90))
    JNT[f'farm_{s}'] = (f'elbow_{s}', (0, -1, 0), (0, 110))
    JNT[f'farm2_{s}'] = (f'forearm_roll_{s}', (0, 0, sg), (-90, 90))
    JNT[f'wrist_{s}'] = (f'wrist_pitch_{s}', (0, -1, 0), (-70, 70))
    JNT[f'flange_{s}'] = (f'wrist_yaw_{s}', (sg, 0, 0), (-40, 40))
PARENT = {'carrier': 'pelvis', 'spider': 'carrier', 'torso': 'spider'}
for s in 'LR':
    PARENT.update({f'shpitch_{s}': 'torso', f'shroll_{s}': f'shpitch_{s}', f'uarm_{s}': f'shroll_{s}', f'farm_{s}': f'uarm_{s}',
                   f'farm2_{s}': f'farm_{s}', f'wrist_{s}': f'farm2_{s}', f'flange_{s}': f'wrist_{s}'})
    PARENT.update({f'hipyaw_{s}': 'pelvis', f'hiproll_{s}': f'hipyaw_{s}', f'thigh_{s}': f'hiproll_{s}', f'shin_{s}': f'thigh_{s}', f'ankspider_{s}': f'shin_{s}', f'foot_{s}': f'ankspider_{s}'})

def members(b):
    if b in ('carrier',): return ['carrier', 'waistbarrel_L', 'waistbarrel_R']
    if b == 'torso': return ['torso', 'waistrod_L', 'waistrod_R', 'pumpshaft', 'pumpring']
    if b in ('pelvis', 'spider'): return [b]
    base, s = b.split('_')
    extra = [e.format(s=s) if '{s}' in e else f'{e}_{s}' for e in LUMP.get(base, [])]
    return [b] + extra
INER = {b: combine(members(b)) for b in J}
tot = sum(v[0] for v in INER.values()); print('total mass', round(tot, 2))

# collision primitives (in world coords at the default pose; converted to body frame)
def capsule(a3, b3, r):
    return dict(type='capsule', fromto=np.concatenate([P(*a3), P(*b3)]), size=[r * U])
COL = {'pelvis': [dict(type='box', pos=P(0, 0.3, -0.2), size=[0.5 * U, 0.75 * U, 0.2 * U])],
       'torso': [dict(type='box', pos=P(0, 3.6, -0.5), size=[1.25 * U, 2.0 * U, 2.0 * U])]}
for s, sg in (('L', 1), ('R', -1)):
    x = sg * lx
    COL[f'thigh_{s}'] = [capsule((x, LY(-1.05), 0.05), (x, LY(-2.55), 0.05), 0.33)]
    COL[f'shin_{s}'] = [capsule((x, LY(-2.8), 0.05), (x, LY(-4.25), 0.0), 0.26)]
    COL[f'foot_{s}'] = [dict(type='box', pos=P(x, LY(-4.842), 0.25), size=[0.71 * U, 0.32 * U, 0.0533 * U], name=f'sole_{s}')]
    ax = sg * 2.98
    COL[f'uarm_{s}'] = [dict(capsule((ax, 4.75, -0.2), (ax, 3.3, -0.25), 0.24), arm=True)]
    COL[f'farm2_{s}'] = [dict(capsule((ax, 2.6, -0.2), (ax, 1.45, -0.2), 0.2), arm=True)]
    COL[f'flange_{s}'] = [dict(type='box', pos=P(ax, 0.92, -0.2), size=[0.19 * U, 0.19 * U, 0.07 * U], arm=True)]

# ---------------------------------------------------------------- visual meshes
PAL = dict(cast=0x34373c, castMid=0x474a51, anod=0x1c1e22, mach=0xbfc2c6, steel=0xdadde1, bolt=0xa9adb3, brass=0xd1a73a, copper=0xc0662a, rust=0x7a5444,
           rubber=0x151515, loom=0x0f0f10, hv=0xe0601a, tagR=0xc4301e, tagB=0x2c6cc0, tagY=0xe0b820, pcb=0x1d5e34, chip=0x0d0d0d, label=0xd9d9d9,
           gear=0xb8bcc2, org=0x6a6f76, pinion=0xd8b46a, oil=0xe88a00, glass=0xcfe6ff, line=0x161616)
def rgba(k):
    v = PAL.get(k, 0x777777); a = 0.12 if k == 'glass' else (0.22 if k == 'oil' else 1)
    return f'{((v >> 16) & 255) / 255:.3f} {((v >> 8) & 255) / 255:.3f} {(v & 255) / 255:.3f} {a}'
VISMAP = {}       # sim body (or mocap part) -> list of (meshname, material)
MOCAP_PARTS = ['tribarrel', 'trirod', 'rocker', 'coupler', 'quadbarrel', 'quadrod', 'achbarrel_{s}0', 'achbarrel_{s}1', 'achrod_{s}0', 'achrod_{s}1']
def part_origin(part):   # reference point the mocap part pose is defined about (world, default pose)
    s = part[-1] if part[-1] in 'LR' else part[-2]
    sg = 1 if s == 'L' else -1; x = sg * lx
    K = np.array([-2.65 - DT, 0.05])
    def k2(p): return P(x, K[0] + p[0], K[1] + p[1])
    if part.startswith('rocker'): return k2([KN['yq'], KN['zq']])
    if part.startswith('coupler'): return k2([KN['yq'] + KN['r2'] * np.cos(KN['phi0']), KN['zq'] + KN['r2'] * np.sin(KN['phi0'])])
    if part.startswith('quadbarrel') or part.startswith('quadrod'): return k2([KN['yb'], KN['zb']])
    if part.startswith('achbarrel') or part.startswith('achrod'):
        i = int(part[-1]); xo = (-0.11 if i == 0 else 0.11)
        return P(x + xo, LY(-2.96), -0.3)
    if part.startswith('waist'): sx = 1 if part.endswith('L') else -1; return P(sx * 0.62, 0.84, -0.62)
    if part == 'pumpshaft': return P(1.15, 4.0, 0.634)
    if part == 'pumpring': return P(1.15, 4.0 - 0.0967742 * 0.62, 0.634)
    if part.startswith('tri'): sx = 1 if part.endswith('L') else -1; return P(sx * 2.98, 4.55, -0.55)
    raise KeyError(part)
def export_meshes(src, origin, budget):
    files = [f for f in os.listdir(f'{HERE}/out/meshes') if f.startswith(src + '__')]
    out = []
    for f in sorted(files):
        mat = f.split('__')[1][:-4]
        mesh = trimesh.load(f'{HERE}/out/meshes/{f}', process=False)
        V = np.asarray(mesh.vertices) + np.array([0, 0, ZG]) - origin; F = np.asarray(mesh.faces)
        if len(F) < 4: continue
        # no decimation: simplification deleted whole thin-walled parts (e.g. accumulator shells).
        # Oversized meshes are split into chunks of <= budget faces instead.
        chunks = [F[i:i + budget] for i in range(0, len(F), budget)]
        for k, Fc in enumerate(chunks):
            used = np.unique(Fc); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
            name = f'{src}__{mat}' + (f'_{k}' if k else '')
            trimesh.Trimesh(V[used], remap[Fc], process=False).export(f'{OUTD}/meshes/{name}.stl')
            out.append((name, mat))
    return out

# ---------------------------------------------------------------- xml
def f3(v): return ' '.join(f'{x:.5f}' for x in v)
def fullinertia(I): return f'{I[0,0]:.6f} {I[1,1]:.6f} {I[2,2]:.6f} {I[0,1]:.6f} {I[0,2]:.6f} {I[1,2]:.6f}'
def body_xml(b, viz, ind=2):
    pad = ' ' * ind; pos = J[b] - (J[PARENT[b]] if b in PARENT else 0)
    m, c, I = INER[b]
    x = [f'{pad}<body name="{b}" pos="{f3(pos)}">']
    if b == 'pelvis': x.append(f'{pad}  <freejoint name="root"/>')
    else:
        jn, ax, rg = JNT[b]
        x.append(f'{pad}  <joint name="{jn}" axis="{f3(ax)}" range="{rg[0]} {rg[1]}" class="hj"/>')
    x.append(f'{pad}  <inertial pos="{f3(c - J[b])}" mass="{m:.4f}" fullinertia="{fullinertia(I)}"/>')
    for g in COL.get(b, []):
        nm = f' name="{g["name"]}"' if 'name' in g else ''
        cls = 'arm' if g.get('arm') else ('sole' if 'sole' in nm else 'col')
        if not viz and cls != 'sole': continue          # training model: only the soles touch anything (MJX contact cost)
        if g['type'] == 'capsule': x.append(f'{pad}  <geom{nm} type="capsule" fromto="{f3(g["fromto"][:3] - J[b])} {f3(g["fromto"][3:] - J[b])}" size="{g["size"][0]:.4f}" class="{cls}"/>')
        else: x.append(f'{pad}  <geom{nm} type="box" pos="{f3(g["pos"] - J[b])}" size="{f3(g["size"])}" class="{cls}"/>')
    if b == 'pelvis': x.append(f'{pad}  <site name="imu" pos="0 0 0"/>')
    if b.startswith('foot'): x.append(f'{pad}  <site name="ft_{b[-1]}" pos="0.03 0 -0.0405"/>')
    if viz:
        for (mn, mat) in VISMAP.get(b, []): x.append(f'{pad}  <geom type="mesh" mesh="{mn}" rgba="{rgba(mat)}" class="viz"/>')
    for ch in [k for k, p in PARENT.items() if p == b]: x += body_xml(ch, viz, ind + 2)
    x.append(f'{pad}</body>')
    return x

def build(viz):
    if viz:
        for b in J: VISMAP[b] = export_meshes(b, J[b], 80000)
        for s in 'LR':
            for p in MOCAP_PARTS:
                pn = p.format(s=s) if '{s}' in p else f'{p}_{s}'
                VISMAP[pn] = export_meshes(pn, part_origin(pn), 80000)
        for p in ('pumpshaft', 'pumpring'): VISMAP[p] = export_meshes(p, part_origin(p), 80000)
        for s in 'LR':
            for p in ('waistbarrel', 'waistrod'):
                VISMAP[f'{p}_{s}'] = export_meshes(f'{p}_{s}', part_origin(f'{p}_{s}'), 80000)
    x = ['<mujoco model="hydro_humanoid_v9">',
         '  <compiler angle="degree" inertiafromgeom="false" autolimits="true"/>',
         '  <option timestep="0.001" iterations="2" ls_iterations="4" integrator="Euler"><flag eulerdamp="disable"/></option>',
         '  <default>',
         '    <joint damping="0.5" armature="0.02" frictionloss="0.2"/>',
         '    <default class="hj"><joint limited="true" solreflimit="0.004 1"/></default>',
         '    <default class="arm"><geom contype="4" conaffinity="1" condim="3" friction="0.8 0.02 0.01" rgba="0.3 0.5 0.9 0.25" group="3"/></default>',
         '    <default class="col"><geom contype="2" conaffinity="5" condim="3" friction="0.9 0.02 0.01" rgba="0.8 0.3 0.2 0.25" group="3"/></default>',
         '    <default class="sole"><geom contype="2" conaffinity="1" condim="3" friction="1.0 0.02 0.01" rgba="0.2 0.8 0.3 0.4" group="3"/></default>',
         '    <default class="viz"><geom contype="0" conaffinity="0" group="1" mass="0"/></default>',
         '  </default>',
         '  <visual><global offwidth="1600" offheight="1200"/><headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/><quality shadowsize="4096"/></visual>',
         '  <asset>', '    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.17 0.19" rgb2="0.12 0.13 0.14" width="512" height="512"/>',
         '    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>']
    if viz:
        for lst in VISMAP.values():
            for mn, _ in lst: x.append(f'    <mesh name="{mn}" file="meshes/{mn}.stl" inertia="shell"/>')
    x += ['  </asset>', '  <worldbody>', '    <light pos="2 -2 4" dir="-1 1 -2" directional="true" castshadow="true"/>',
          '    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1" friction="1 0.02 0.01"/>']
    x += body_xml('pelvis', viz, 4)
    if viz:   # moving linkage / cylinder parts, posed every frame by viewer.py from the exact kinematics
        for nm, lst in VISMAP.items():
            if nm in J: continue
            x.append(f'    <body name="mc_{nm}" mocap="true">')
            for (mn, mat) in lst: x.append(f'      <geom type="mesh" mesh="{mn}" rgba="{rgba(mat)}" class="viz"/>')
            x.append('    </body>')
    x += ['  </worldbody>']
    if viz:   # zero-gain 'actuators' = joint-target sliders in the viewer's Control panel (degrees). They apply NO force:
              # viewer.py reads them as targets for the hydraulic firmware, so the oil does the work.
        x.append('  <actuator>')
        ORDER = ['waist_yaw', 'waist_pitch', 'waist_roll'] + [f'{j}_{s}' for s in 'LR' for j in ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle_pitch', 'ankle_roll')] + \
                [f'{j}_{s}' for s in 'LR' for j in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'forearm_roll', 'wrist_pitch', 'wrist_yaw')]
        RG = {jn: rg for (jn, ax, rg) in JNT.values()}
        for jn in ORDER: x.append(f'    <general name="{jn}" joint="{jn}" gainprm="0" biasprm="0 0 0" ctrlrange="{RG[jn][0]} {RG[jn][1]}"/>')
        x.append('  </actuator>')
    x += ['  <sensor>', '    <gyro name="gyro" site="imu"/>', '    <accelerometer name="accel" site="imu"/>',
          '    <force name="ft_force_L" site="ft_L"/>', '    <torque name="ft_torque_L" site="ft_L"/>', '    <force name="ft_force_R" site="ft_R"/>', '    <torque name="ft_torque_R" site="ft_R"/>',
          '  </sensor>', '</mujoco>']
    fn = f'{OUTD}/hydro_humanoid{"_viz" if viz else ""}.xml'; open(fn, 'w').write('\n'.join(x)); print('wrote', fn)

build(False); build(True)
json.dump({k: [float(a) for a in v] for k, v in J.items()}, open(f'{OUTD}/joint_points.json', 'w'), indent=1)
