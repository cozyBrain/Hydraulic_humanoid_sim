import sys; sys.path.insert(0,'/home/claude/tripo/sim')
import numpy as np, mujoco, sim_core as sc, imageio
import viz_util as vu
m=mujoco.MjModel.from_xml_path('model/hydro_humanoid_viz.xml'); d=mujoco.MjData(m)
q0,_=sc.default_q(*sc.load('model/hydro_humanoid_viz.xml')[0:2]); qadr=np.array(sc.load('model/hydro_humanoid_viz.xml')[1])
R=np.load(sys.argv[1]); tab=R['table']; idx=R['idx']; root=R['root']; zr=float(R['z_ref']); f=float(R['f_ref']); v=float(R['v_ref']); n=len(tab)
out=sys.argv[2]; T=float(sys.argv[3]); az=float(sys.argv[4]) if len(sys.argv)>4 else 90
sole=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f'sole_{s}') for s in 'LR']
r=mujoco.Renderer(m,480,640); cam=mujoco.MjvCamera(); cam.distance=3.2; cam.elevation=-8; cam.azimuth=az
poser=vu.LinkagePoser(m)
def Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def pose(ph, x0, off):
    i=ph*(n-1); i0=int(i); a=i-i0; i1=min(i0+1,n-1)
    q=tab[i0]*(1-a)+tab[i1]*a; roll,pitch_up,yaw,lat,up=root[i0]*(1-a)+root[i1]*a
    qp=q0.copy(); qp[qadr[idx]]=q; qp[0]=x0; qp[1]=lat; qp[2]=zr+up-off
    quat=np.zeros(4); mujoco.mju_mat2Quat(quat,(Rz(yaw)@Ry(-pitch_up)@Rx(roll)).flatten()); qp[3:7]=quat
    d.qpos[:]=qp; mujoco.mj_forward(m,d)
    return min(min((d.geom_xpos[g]+d.geom_xmat[g].reshape(3,3)@(np.array([sx,sy,-1.0])*m.geom_size[g]))[2] for sx in(-1,1) for sy in(-1,1)) for g in sole)
# ground contact: lowest sole height over the cycle, smoothed (no jerky snapping)
lows=np.array([pose(p,0,0) for p in np.arange(100)/100]); k=6
lows_s=np.array([np.mean(np.take(lows,range(i-k,i+k+1),mode='wrap')) for i in range(100)])
frames=[]; LOW=[]; fps=30
for j in range(int(T*fps)):
    t=j/fps; ph=(f*t)%1.0; off=np.interp(ph*100,np.arange(101),np.append(lows_s,lows_s[0]))
    LOW.append(pose(ph,v*t,off)); poser(d,0.0)
    cam.lookat[:]=[v*t,0,0.9]; r.update_scene(d,cam); frames.append(r.render())
imageio.mimsave(out,frames,fps=fps,quality=8); print('ok',len(frames),'lowest sole min %.3f max %.3f'%(min(LOW),max(LOW)))
