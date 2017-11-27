
import numpy as np

from GPS_INS.GPS_INS_ECI import GPSINS_ECI_errormodel

Rb_imu = np.array([[0],
                        [0],
                        [0]])

Cb_imu = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

IMU_ac_Wn = np.array([[0],
                        [0],
                        [0]])

IMU_gy_Wn = np.array([[0],
                        [0],
                        [0]])

GPS_po_Wn = np.array([[0],
                        [0],
                        [0]])

GpsIns = GPSINS_ECI_errormodel(Rb_imu, Cb_imu, IMU_ac_Wn, IMU_gy_Wn, GPS_po_Wn)

r_e = np.array([[0/180*np.pi],
                [10/180*np.pi],
                [0]])

v_e = np.array([[0],
                [10],
                [0]])

qbe = np.array([[1],
                [0],
                [0],
                [0]])

[r_i, v_i, qbi] = GpsIns.ENUtoECI(r_e, v_e, qbe)

print(r_i, v_i, qbi)

"""
r_i = np.array([[6.3781 * 10**6],
                [0],
                [0]])

v_i = np.array([[0],
                [0],
                [0]])

qbi = np.array([[1],
                [0],
                [0],
                [0]])
"""


aBias = np.array([[0],
                  [0],
                  [0]])

wBias = np.array([[0],
                  [0],
                  [0]])

X_ins0 = np.r_[r_i, v_i, qbi, aBias, wBias]
print("X_ins0 = \n", X_ins0)

a = np.array([[9.80665],
              [0],
              [0]])

w = np.array([[0.0],
              [0],
              [0]])

dt = 0.02

X_ins1 = GpsIns.INSupdate(a, w, X_ins0, dt)

print("X_ins1 = \n", X_ins1)