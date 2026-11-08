"""Leg-length parameters shared by the CAD extraction, the MJCF builder and the kinematics (v10: legs +20 cm).
Design units (1 u = 0.15 m), three.js frame.  LY maps a v9 leg y coordinate to the v10 one (same function as robot24.html)."""
DT, DS = 0.6667, 0.6667                      # thigh / shin extension (u)  -> +10 cm each
def LY(y):
    if y >= -0.94: return y
    if y > -2.65: return -0.94 + (y + 0.94) * (1.71 + DT) / 1.71
    if y >= -3.05: return y - DT
    if y > -4.22: return -3.05 - DT + (y + 3.05) * (1.17 + DS) / 1.17
    return y - DT - DS
U = 0.15
ZG = (4.895 + DT + DS) * U                   # sole bottom at three y = LY(-4.895)
