"""Split the v9 model (GLB, knee 0 deg) into simulation bodies.
For every body: mass, COM, inertia (from mesh volumes x material density) and per-material visual meshes.
Output coordinates: MuJoCo frame, metres.  mj = (three.z, three.x, three.y) * 0.15
usage: python3 extract_bodies.py -- model.glb ctr.json lines.json outdir"""
import bpy, sys, os, json, math, re
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
argv = sys.argv[sys.argv.index('--') + 1:]
GLB, CTR, LINES, OUT = argv
os.makedirs(OUT + '/meshes', exist_ok=True)
ctr = np.array(json.load(open(CTR)))
U = 0.15
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from legs import LY
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=GLB)
dg = bpy.context.evaluated_depsgraph_get()

# ---- material classes by base colour (three.js palette, sRGB hex) -> density kg/m^3
PAL = dict(cast=0x34373c, castMid=0x474a51, anod=0x1c1e22, mach=0xbfc2c6, steel=0xdadde1, bolt=0xa9adb3, brass=0xd1a73a, copper=0xc0662a,
           rust=0x7a5444, rubber=0x151515, loom=0x0f0f10, hv=0xe0601a, tagR=0xc4301e, tagB=0x2c6cc0, tagY=0xe0b820, pcb=0x1d5e34, chip=0x0d0d0d,
           label=0xd9d9d9, gear=0xb8bcc2, org=0x6a6f76, pinion=0xd8b46a, oil=0xe88a00)
def s2l(c): c = c / 255.0; return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
PALV = {k: s2l(np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255], float)) for k, v in PAL.items()}
# effective densities: structural parts are modelled solid but are hollow/pocketed in reality
DENS = dict(cast=2700 * 0.45, castMid=2700 * 0.6, anod=2700 * 0.35, mach=2700 * 0.6, steel=7850 * 0.9, bolt=7850, brass=8500 * 0.8, copper=8900,
            rust=2700, rubber=1150, loom=600, hv=1500, tagR=1200, tagB=1200, tagY=1200, pcb=1850, chip=1500, label=1200, gear=7850, org=4430 * 0.22,
            pinion=7850, oil=870, glass=1200)
def matclass(o):
    m = o.active_material
    if m is None: return 'cast'
    if o.name.startswith('glass'): return 'glass'
    if o.name.startswith('oil'): return 'oil'
    try:
        bc = np.array(m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value[:3])
    except Exception: return 'cast'
    k = min(PALV, key=lambda k: np.sum((PALV[k] - bc) ** 2))
    return k

def T3(p):  # blender world -> three (original, uncentred)
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1) + ctr
def MJ(p3):  # three -> mujoco metres
    return np.stack([p3[:, 2], p3[:, 0], p3[:, 1]], 1) * U

# ---- body classifier (three.js coordinates, design units)
def leg_side(x): return 'L' if x > 0 else 'R'
def classify(name, c, group, gproj):
    x, y, z = c
    m = re.match(r'^(thigh|shin|rock|cpl|qcb|qcr)([LR])_', name)
    if m and m.group(1) == 'thigh' and y > -0.9: m = None          # thigh-tagged line ends that sit on the pelvis / hip housings
    if group is not None:                    # inside an hcyl group: barrel vs rod by axial position
        kind, side, t, bl = group
        part = 'rod' if (t > bl + 0.1) else 'barrel'
        if kind == 'ach': return f'ach{part}_{side}{0 if x - (0.85 if side == "L" else -0.85) < 0 else 1}'
        if kind == 'waist': return f'waist{part}_{"L" if x > 0 else "R"}'
        if kind == 'tri': return f'tri{part}_{"L" if x > 0 else "R"}'
    if m:
        tag, sd = m.group(1), m.group(2)
        if tag == 'qcb': return 'quadbarrel_' + sd
        if tag == 'qcr': return 'quadrod_' + sd
        if tag == 'rock': return 'rocker_' + sd
        if tag == 'cpl': return 'coupler_' + sd
        if tag == 'thigh': return 'thigh_' + sd
        lx = 0.85 if sd == 'L' else -0.85
        if abs(x - lx) < 0.3 and LY(-4.72) < y < LY(-4.28) and abs(z - 0.05) < 0.3 and y > LY(-4.6):
            dy = y - LY(-4.5)
            if dy > 0.06: return 'shin_' + sd
            if dy < -0.06: return 'foot_' + sd
            dx, dz = abs(x - lx), abs(z - 0.05)
            if dx < 0.06 and dz < 0.06: return 'ankspider_' + sd
            return ('foot_' if dz > dx else 'shin_') + sd
        if y < LY(-4.56): return 'foot_' + sd
        return 'shin_' + sd
    # rotating pump parts (visual mocap bodies; their mass stays with the torso in the dynamics)
    base = name.split('.')[0]
    if base in ('spin_shaft', 'spin_hub1', 'spin_spider', 'spin_hub2', 'spin_pinion', 'spin_nut'): return 'pumpshaft'
    if base == 'ring_gear': return 'pumpring'
    # untagged
    if abs(x) > 2.67 and y > 0.5:          # arm chain (shoulder pitch housing itself stays on the torso)
        sd = 'L' if x > 0 else 'R'
        if y >= 5.09: return 'shpitch_' + sd
        if y >= 4.77: return 'shroll_' + sd
        if 3.05 < y < 3.4 and z < -0.3: return 'farm_' + sd            # olecranon horn + rod-end block ride on the forearm
        if y >= 3.05: return 'uarm_' + sd
        if y > 2.64: return 'farm_' + sd
        if y > 1.36: return 'farm2_' + sd
        if y > 1.05: return 'wrist_' + sd
        return 'flange_' + sd
    if abs(x) > 0.3 and y < 0.2:
        sd = leg_side(x)
        if y > -0.05: return 'pelvis'
        if y > -0.5: return 'hipyaw_' + sd
        if y > -1.05: return 'hiproll_' + sd
        if y > LY(-2.6): return 'thigh_' + sd
        if y > LY(-4.56): return 'shin_' + sd
        return 'foot_' + sd
    if abs(x) < 0.5 and 0.69 < y < 1.5 and abs(z + 0.2) < 0.5:
        if y < 1.0: return 'carrier'
        if y > 1.2: return 'torso'
        dx, dz = abs(x), abs(z + 0.2)
        if dx < 0.06 and dz < 0.06: return 'spider'
        return 'carrier' if dx > dz else 'torso'
    if y < 0.69: return 'pelvis'
    return 'torso'

# ---- hcyl group detection
groups = {}
for o in bpy.data.objects:
    if o.type == 'EMPTY' and ('achilles_cylinder' in o.name or o.name.startswith('waist_cylinder') or o.name.startswith('triceps_cylinder')):
        kids = [k for k in o.children if k.type == 'MESH']
        org = T3(np.array([list(o.matrix_world.translation)]))[0]
        rod = [k for k in kids if '_rod' in k.name]
        cen = lambda k: T3(np.array([list(k.matrix_world @ v.co) for v in k.data.vertices])).mean(0)
        d = cen(rod[0]) - org if rod else None
        if d is None: continue
        d = d / np.linalg.norm(d)
        ts = {k.name: float((cen(k) - org) @ d) for k in kids}
        L = max(ts.values()) + 0.03
        side = ('L' if 'shinL' in o.name else 'R') if 'achilles' in o.name else ('L' if org[0] > 0 else 'R')
        kind = 'ach' if 'achilles' in o.name else ('tri' if 'triceps' in o.name else 'waist')
        for k in kids: groups[k.name] = (kind, side, ts[k.name], 0.6 * L)

# ---- tetra mass properties
def massprops(V, F, rho):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    v6 = np.einsum('ij,ij->i', a, np.cross(b, c))           # 6 * signed tetra volume (origin apex)
    vol = v6.sum() / 6
    if abs(vol) < 1e-12: return 0, np.zeros(3), np.zeros((3, 3)), 0
    sgn = np.sign(vol); v6 = v6 * sgn; vol = abs(vol)
    com = ((a + b + c) * v6[:, None]).sum(0) / 24 / vol
    # second moments about origin: int x_i x_j dV
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = (v6 * (2 * a[:, i] * a[:, j] + 2 * b[:, i] * b[:, j] + 2 * c[:, i] * c[:, j] + a[:, i] * b[:, j] + a[:, j] * b[:, i]
                            + a[:, i] * c[:, j] + a[:, j] * c[:, i] + b[:, i] * c[:, j] + b[:, j] * c[:, i])).sum() / 120
    m = rho * vol; S = S * rho
    I = np.trace(S) * np.eye(3) - S                      # about origin
    I = I - m * (np.dot(com, com) * np.eye(3) - np.outer(com, com))   # shift to COM
    return m, com, I, vol

LINESJ = {L['id']: L for L in json.load(open(LINES))}
# mass overrides: EMRAX 188 LV liquid-cooled 7.9 kg (datasheet); 1 L piston accumulator shells 2.2 kg each (aluminium body, 210 bar)
MASS_FIX = {'emrax_': 7.9, 'accumulator': 2.2 * 2}
FIXVOL = {'emrax_': 0.0, 'accumulator': 0.0}
for o in bpy.data.objects:
    if o.type != 'MESH': continue
    base = o.name.split('.')[0]
    key = 'emrax_' if base.startswith('emrax_') else ('accumulator' if base == 'accumulator' else None)
    if key is None: continue
    me = o.evaluated_get(dg).to_mesh(); me.calc_loop_triangles()
    Vb = np.array([list(o.matrix_world @ v.co) for v in me.vertices]); F = np.array([list(t.vertices) for t in me.loop_triangles])
    FIXVOL[key] += massprops(MJ(T3(Vb)), F, 1.0)[3]
print('mass-fix volumes', FIXVOL)
bodies = {}
BIG = []
BYC = {}
def addto(b, mcls, m, com, I, V, F):
    BYC[(b, mcls)] = BYC.get((b, mcls), 0) + m
    B = bodies.setdefault(b, dict(m=0.0, mc=np.zeros(3), parts=[], vis={}))
    B['parts'].append((m, com, I)); B['m'] += m; B['mc'] += m * com
    if V is not None:
        vl = B['vis'].setdefault(mcls, ([], [], [0]))
        vl[0].append(V); vl[1].append(F + vl[2][0]); vl[2][0] += len(V)

for o in bpy.data.objects:
    if o.type != 'MESH': continue
    me = o.evaluated_get(dg).to_mesh(); me.calc_loop_triangles()
    if not len(me.loop_triangles): continue
    Vb = np.array([list(o.matrix_world @ v.co) for v in me.vertices]); F = np.array([list(t.vertices) for t in me.loop_triangles])
    V3 = T3(Vb); Vm = MJ(V3)
    n = o.name; mcls = matclass(o)
    isline = n.startswith('L') and n[1:].split('_')[0] in ('hose', 'loom', 'hard', 'bundle', 'twin')
    # split merged detail meshes into loose parts, classify each
    if 'detail' in n or isline:
        q = np.round(V3 / 1e-4).astype(np.int64)
        _, inv = np.unique(q, axis=0, return_inverse=True); inv = inv.ravel()
        uniq = np.zeros((inv.max() + 1, 3)); uniq[inv] = V3; V3 = uniq
        uniqm = np.zeros((inv.max() + 1, 3)); uniqm[inv] = Vm; Vm = uniqm
        F = inv[F]
        g = coo_matrix((np.ones(len(F) * 3), (np.repeat(F[:, 0], 3), F.ravel())), shape=(len(V3), len(V3)))
        nc, lab = connected_components(g + g.T, directed=False)
        flab = lab[F[:, 0]]
        tag = re.match(r'^(thigh|shin|rock|cpl|qcb|qcr)([LR])_', n)
        for ci in range(nc):
            fi = F[flab == ci]
            if not len(fi): continue
            vi = np.unique(fi); remap = -np.ones(len(V3), int); remap[vi] = np.arange(len(vi))
            c3 = V3[vi].mean(0)
            b = classify((tag.group(0) if tag else '') + 'x', c3, None, None)
            if isline:
                lid = n[1:].split('.')[0]; L = LINESJ.get(lid)
                # hose/tube with oil: mass from length (tube geometry is open)
                pts = np.array(L['pts']) + 0 if L else None
                length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) * U if L else 0
                nl = L.get('n', 1) if L else 1
                lin = {'hose': 0.35, 'hard': 0.45, 'loom': 0.12, 'bundle': 0.3, 'twin': 0.3}[n[1:].split('_')[0]]
                m = length * lin * (nl if n[1:].startswith('bundle') else (2 if n[1:].startswith('twin') else 1)) / max(nc, 1)
                com = Vm[vi].mean(0); I = np.eye(3) * 1e-6
            else:
                m, com, I, vol = massprops(Vm[vi], remap[fi], DENS[mcls])
                if m == 0: com = Vm[vi].mean(0); I = np.eye(3) * 1e-9
            addto(b, 'line' if isline else mcls, m, com, I, Vm[vi], remap[fi])
        continue
    base = n.split('.')[0]
    rho = DENS.get(mcls, 2000)
    if base.startswith('emrax_'): rho = MASS_FIX['emrax_'] / max(FIXVOL['emrax_'], 1e-12)          # real motor mass, spread over its parts
    if base == 'accumulator': rho = MASS_FIX['accumulator'] / max(FIXVOL['accumulator'], 1e-12)
    m, com, I, vol = massprops(Vm, F, rho)
    if m == 0: com = Vm.mean(0); I = np.eye(3) * 1e-9
    c3 = V3.mean(0)
    b = classify(n, c3, groups.get(n), None)
    addto(b, mcls, m, com, I, Vm, F)
    BIG.append((m, n, mcls, b, vol))

# battery cells: the battery box is modelled as a thin shell 'cast' -> add the pack mass explicitly (3.5 kWh LFP-class ~ 18 kg)
out = {}
for b, B in bodies.items():
    m = B['m']; com = B['mc'] / max(m, 1e-9)
    I = np.zeros((3, 3))
    for (mi, ci, Ii) in B['parts']:
        d = ci - com; I += Ii + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    out[b] = dict(mass=float(m), com=com.tolist(), inertia=I.tolist())
    for mcls, (Vs, Fs, _) in B['vis'].items():
        V = np.concatenate(Vs); F = np.concatenate(Fs)
        with open(f'{OUT}/meshes/{b}__{mcls}.obj', 'w') as f:
            f.write(''.join(f'v {a:.5f} {b_:.5f} {c:.5f}\n' for a, b_, c in V)); f.write(''.join(f'f {i + 1} {j + 1} {k + 1}\n' for i, j, k in F))
json.dump(out, open(f'{OUT}/bodies_raw.json', 'w'), indent=1)
for k_, v_ in sorted(BYC.items(), key=lambda r: -r[1])[:15]: print('BYC', k_, round(v_, 2))
for r in sorted(BIG, key=lambda r: -r[0])[:30]: print('BIG', round(r[0],2), r[1], r[2], r[3], round(r[4]*1e3,3), 'L')
tot = sum(v['mass'] for v in out.values())
for b in sorted(out): print(f"{b:16s} {out[b]['mass']:8.2f} kg  com {np.round(out[b]['com'], 3)}")
print('TOTAL', round(tot, 2))
