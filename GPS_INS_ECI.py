# -*- coding: utf-8 -*-

"""
_e : ENU frame
_b : body frame (origin is )
_i : Inartial frame (ECI)


X_ins = [r_i v_i q_bi aBias wBias]T 0-2 3-5 6-9 10-12 13-15


参考にした論文
システムモデル：GPS/INS Kalman Filter Design for Spacecraft Operating in the Proximity of International Space Station
連続時間システムから離散時間システムへの式展開方法：搬送波位相DGPS/INS 複合航法アルゴリズムの開発
フィルタの試験方法（予定）：Assesment of integrated GPS/INS for the EX-171 Extended Range Guided Munition
"""

import numpy as np

class GPSINS_ECI_errormodel:
    # 地球の重力・形の定数
    Mu = 3.986004418 * 10 ** (14)
    f = 1/298.257222101
    Re = 6.378137 * 10 ** 6
    J2 = 108263 * 10 ** (-8)
    OmeE = np.array([[0], [0], [7.292*10**(-5)]])

    def __init__(self, Tau_accel, Tau_gyro, IMU_ac_Wn, IMU_gy_Wn, GPS_r_Wn, GPS_v_Wn):
        # オブジェクト作成時に設定するパラメータ．
        # センサ位置，分散値等は最初に設定したらその後は固定
        self.Tau_a = Tau_accel              # 加速度センサのcorrelation times
        self.Tau_g = Tau_gyro               # 角速度センサのcorrelation times
        self.Wno_a = IMU_ac_Wn              # 加速度センサの分散値（ホワイトノイズによる分散）
        self.Wno_g = IMU_gy_Wn              # 角速度センサの分散値（ホワイトノイズによる分散）
        self.WnGp_r = GPS_r_Wn              # GPSでの位置分散値（ホワイトノイズによる分散）
        self.WnGp_v = GPS_v_Wn              # GPSでの速度分散値（ホワイトノイズによる分散）

    def INSupdate(self, a, w, X_ins, dt):
        # 慣性航法による時間更新

        r_i = np.array([[X_ins[0, 0]], [X_ins[1, 0]], [X_ins[2, 0]]])
        v_i = np.array([[X_ins[3, 0]], [X_ins[4, 0]], [X_ins[5, 0]]])
        qbi = np.array([[X_ins[6, 0]], [X_ins[7, 0]], [X_ins[8, 0]], [X_ins[9, 0]]])
        wBias = np.array([[X_ins[10, 0]], [X_ins[11, 0]], [X_ins[12, 0]]])
        aBias = np.array([[X_ins[13, 0]], [X_ins[14, 0]], [X_ins[15, 0]]])

        w_b = w - wBias
        a_b = a - aBias

        Cbi = self.Quate_ToDCM(qbi)
        a_i = np.dot(Cbi, a_b)
        g = self.Gravity(r_i)

        wx = w_b[0, 0]
        wy = w_b[1, 0]
        wz = w_b[2, 0]
        Ome = 0.5*np.array([[0, wz, -wy, wx],
                        [-wz, 0, wx, wy],
                        [wy, -wx, 0, wz],
                        [-wx, -wy, -wz, 0]])

        qbi1 = qbi + dt * np.dot(Ome, qbi)
        v_i1 = v_i + dt*(a_i + g)
        r_i1 = r_i + dt*v_i

        X_ins1 = np.r_[r_i1, v_i1, qbi1, aBias, wBias]
        return X_ins1

    def KF_predict(self, a, w, X_ins, P_k0_me, dt_pre):
        # カルマンフィルタ予測更新
        # Phi,Qの具体的な算出には，『搬送波位相DGPS/INS 複合航法アルゴリズムの開発』を参考にした．URL:https://repository.exst.jaxa.jp/dspace/handle/a-is/32573

        Phi = self.KF_TimeTransMat(a, w, X_ins, dt_pre)
        Q = self.NoiseCovMat(X_ins, dt_pre)

        P_k1_pr = np.dot(Phi, np.dot(P_k0_me, Phi.T)) + Q

        return P_k1_pr

    def KF_measurement(self, X_ins, P_k1_pr, r_i_gps, v_i_gps):
        # カルマンフィルタ観測更新

        r_i = np.array([[X_ins[0, 0]], [X_ins[1, 0]], [X_ins[2, 0]]])
        v_i = np.array([[X_ins[3, 0]], [X_ins[4, 0]], [X_ins[5, 0]]])

        # 残差計算
        d_r_i = r_i_gps - r_i
        d_v_i = v_i_gps - v_i
        y = np.r_(d_r_i, d_v_i)

        C = np.eye(6, 16)
        R = self.MeasureNoiCovMat()

        # カルマンゲインの算出
        PC_T = np.dot(P_k1_pr, C.T)
        invCPC_R = np.linalg.inv(np.dot(C, PC_T) + R)
        KalG = np.dot(PC_T, invCPC_R)

        # 誤差を推定
        X_k1_pl = np.dot(KalG, y)   # X^が基本常に0なので簡素化される

        # Pを算出
        P_k1_me = np.dot(np.eye(16) - np.dot(KalG, C), P_k1_pr)

        X_ins = X_ins + X_k1_pl
        return X_ins, KalG, P_k1_me

    def KF_TimeTransMat(self, a, w, X_ins, T):
        # カルマンフィルタの状態遷移行列の算出
        # 一時近似で離散化

        F = self.KF_System(a, w, X_ins)
        I_16 = np.eye(16)

        Phi = I_16 + T*F
        Phi[10:12, 10:12] = np.exp(- T / self.Tau_g) * np.eye(3)
        Phi[13:15, 13:15] = np.exp(- T / self.Tau_a) * np.eye(3)

        return Phi

    def KF_System(self, a, w, X_ins):
        # カルマンフィルタのシステム行列算出
        # Fは，INS誤差のシステム行列

        r_i = np.array([[X_ins[0, 0]], [X_ins[1, 0]], [X_ins[2, 0]]])
        v_i = np.array([[X_ins[3, 0]], [X_ins[4, 0]], [X_ins[5, 0]]])
        qbi = np.array([[X_ins[6, 0]], [X_ins[7, 0]], [X_ins[8, 0]], [X_ins[9, 0]]])
        wBias = np.array([[X_ins[10, 0]], [X_ins[11, 0]], [X_ins[12, 0]]])
        aBias = np.array([[X_ins[13, 0]], [X_ins[14, 0]], [X_ins[15, 0]]])

        w_b = w - wBias
        a_b = a - aBias

        r = np.linalg.norm(r_i)
        x = r_i[0, 0]
        y = r_i[1, 0]
        z = r_i[2, 0]

        q1 = qbi[0, 0]
        q2 = qbi[1, 0]
        q3 = qbi[2, 0]
        q4 = qbi[3, 0]

        wx = w_b[0, 0]
        wy = w_b[1, 0]
        wz = w_b[2, 0]

        ax = a_b[0, 0]
        ay = a_b[1, 0]
        az = a_b[2, 0]

        I_33 = np.eye(3)
        Ze33 = np.zeros(3, 3)
        Ze34 = np.zeros(3, 4)
        Ze43 = np.zeros(4, 3)

        G = 3*self.Mu/(r**5)*np.array([x*x, x*y, x*z], [x*y, y*y, y*z], [x*z, y*z, z*z]) - self.Mu/(r**3)*I_33
        Ome = 0.5 * np.array([[0, wz, -wy, wx],
                              [-wz, 0, wx, wy],
                              [wy, -wx, 0, wz],
                              [-wx, -wy, -wz, 0]])
        Q = 0.5 * np.array([[q4, -q3, q2],
                            [q3, q4, -q1],
                            [-q2, q1, q4],
                            [-q1, -q2, -q3]])
        Rbar = np.c_[2*Q, -qbi]
        Abar = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0], [-ax, -ay, -az]])
        Cbi = self.Quate_ToDCM(qbi)

        D = np.dot(2*Cbi, np.dot(Abar.T, Rbar.T))

        Fbg = - 1 / self.Tau_g * I_33
        Fba = - 1/self.Tau_a*I_33

        F = np.r_(np.c_(Ze33, I_33, Ze34, Ze33, Ze33),
                  np.c_(G, Ze33, D, Ze33, Cbi),
                  np.c_(Ze43, Ze43, Ome, Q, Ze43),
                  np.c_(Ze33, Ze33, Ze34, Fbg, Ze33),
                  np.c_(Ze33, Ze33, Ze34, Ze33, Fba))

        return F

    def NoiseCovMat(self, X_ins, T):
        # プロセスノイズの共分散行列を計算
        # 各成分の式展開は『搬送波位相DGPS/INS 複合航法アルゴリズムの開発』を参考にした．URL:https://repository.exst.jaxa.jp/dspace/handle/a-is/32573

        qbi = np.array([[X_ins[6, 0]], [X_ins[7, 0]], [X_ins[8, 0]], [X_ins[9, 0]]])

        q1 = qbi[0, 0]
        q2 = qbi[1, 0]
        q3 = qbi[2, 0]
        q4 = qbi[3, 0]

        Q = 0.5 * np.array([[q4, -q3, q2],
                            [q3, q4, -q1],
                            [-q2, q1, q4],
                            [-q1, -q2, -q3]])

        Cbi = self.Quate_ToDCM(qbi)

        # 時間積分時の近似値（指数で与えてもほとんど問題ないと思われる．参考論文は90年代のもの）
        T1g = T * (1 - T / self.Tau_g + 2 / 3 * (T / self.Tau_g) ** 2)
        T1a = T * (1 - T / self.Tau_a + 2 / 3 * (T / self.Tau_a) ** 2)
        T2g = T ** 2 * (1 / 2 - 1 / 3 * T / self.Tau_g + 1 / 6 * (T / self.Tau_g) ** 2)
        T2a = T ** 2 * (1 / 2 - 1 / 3 * T / self.Tau_a + 1 / 6 * (T / self.Tau_a) ** 2)
        T3 = 1 / 3 * T ** 3

        I_33 = np.eye(3)
        Ze33 = np.zeros(3, 3)
        Ze34 = np.zeros(3, 4)
        Ze43 = np.zeros(4, 3)

        Q_22 = T3*self.Wno_a*Cbi.T
        Q_25 = T2a*self.Wno_a*Cbi
        Q_33 = T3*self.Wno_g*np.dot(Q, Q.T)
        Q_34 = T2g*self.Wno_g*Q
        Q_43 = T2g*self.Wno_g*Q.T
        Q_44 = T1g*self.Wno_g*I_33
        Q_52 = T2a*self.Wno_a*Cbi.T
        Q_55 = T1a*self.Wno_a*I_33

        Q_noise = np.r_(np.c_(Ze33, Ze33, Ze34, Ze33, Ze33),
                        np.c_(Ze33, Q_22, Ze34, Ze33, Q_25),
                        np.c_(Ze43, Ze43, Q_33, Q_34, Ze43),
                        np.c_(Ze33, Ze33, Q_43, Q_44, Ze33),
                        np.c_(Ze33, Q_52, Ze34, Ze33, Q_55))

        return Q_noise

    def MeasureNoiCovMat(self):
        # 観測ノイズの共分散行列
        R = np.zeros(6, 6)
        R[0, 0] = self.WnGp_r
        R[1, 1] = self.WnGp_r
        R[2, 2] = self.WnGp_r
        R[3, 3] = self.WnGp_v
        R[4, 4] = self.WnGp_v
        R[5, 5] = self.WnGp_v

        return R

    def Quate_ToDCM(self, qbi):
        # クォータニオンからDCMを算出

        q1 = qbi[0, 0]
        q2 = qbi[1, 0]
        q3 = qbi[2, 0]
        q4 = qbi[3, 0]

        Cbi = np.array([[q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4, 2 * (q1 * q2 - q3 * q4), 2 * (q3 * q1 + q2 * q4)],
                       [2 * (q1 * q2 + q3 * q4), q2 * q2 - q1 * q1 - q3 * q3 + q4 * q4, 2 * (q2 * q3 - q1 * q4)],
                       [2 * (q3 * q1 - q2 * q4), 2 * (q2 * q3 + q1 * q4), q3 * q3 - q1 * q1 - q2 * q2 + q4 * q4]])

        return Cbi

    def DCM_ToQuate(self, DCM):
        # DCMからクォータニオンへの変換．q1-q4の定義が前提

        C = DCM
        trC = C[0, 0] + C[1, 1] + C[2, 2]
        q1 = np.sqrt(C[0, 0]/2 + (1 - trC)/4)
        q2 = np.sqrt(C[1, 1] / 2 + (1 - trC) / 4)
        q3 = np.sqrt(C[2, 2] / 2 + (1 - trC) / 4)
        q4 = np.sqrt(1 + trC)/2

        q1_abs = np.abs(q1)
        q2_abs = np.abs(q2)
        q3_abs = np.abs(q3)

        if q1_abs>=q2_abs and q1_abs>=q3_abs:
            qAlpha = q1_abs
            qBeta = q2_abs
            qGamma = q3_abs

            Alpha = 1
            Beta = 2
            Gamma = 3
        elif q2_abs>=q3_abs and q2_abs>=q1_abs:
            qAlpha = q2_abs
            qBeta = q3_abs
            qGamma = q1_abs

            Alpha = 2
            Beta = 3
            Gamma = 1
        elif q3_abs>=q1_abs and q3_abs>=q2_abs:
            qAlpha = q3_abs
            qBeta = q1_abs
            qGamma = q2_abs

            Alpha = 3
            Beta = 1
            Gamma = 2
        else:
            print("DCM ERROR-'DCM_ToQuate'")

        if q4 != 0:
            qAlpha1 = np.sign(C[Gamma - 1][Beta - 1] - C[Beta - 1][Gamma - 1]) * qAlpha
            qBeta1 = np.sign(qAlpha * (C[Beta - 1][Alpha - 1] + C[Alpha - 1][Beta - 1])) * qBeta
            qGamma1 = np.sign(qAlpha * (C[Gamma - 1][Alpha - 1] + C[Alpha - 1][Gamma - 1])) * qGamma
        else:
            qAlpha1 = qAlpha
            qBeta1 = np.sign(qAlpha * (C[Beta - 1][Alpha - 1] + C[Alpha - 1][Beta - 1])) * qBeta
            qGamma1 = np.sign(qAlpha * (C[Gamma - 1][Alpha - 1] + C[Alpha - 1][Gamma - 1])) * qGamma

        q = np.zeros([4, 1])
        q[Alpha - 1, 0] = qAlpha1
        q[Beta - 1, 0] = qBeta1
        q[Gamma - 1, 0] = qGamma1
        q[3, 0] = q4
        # 正規化はあえて行わない．あり得ないDCMの場合は出力のノルムが1以外になる
        return q

    def Gravity(self, r_i):
        # ECI座標系での位置データから，ECI座標系での重力ベクトルを計算
        # J2項まで考慮

        rx = r_i[0, 0]
        ry = r_i[1, 0]
        rz = r_i[2, 0]

        r = np.sqrt(rx*rx + ry*ry + rz*rz)
        r_xy = np.sqrt(rx*rx + ry*ry)

        phi = np.arctan2(r_xy, rz) # phiは北極が0degの地心緯度，0<phi<180deg

        Gr = - self.Mu/r**2 * (1 - 3/2*self.J2*(self.Re/r)**2 * (3*np.cos(phi)**2 - 1))    # 地球中心方向の重力ベクトルの大きさ
        Gp = 3*self.Mu/r**2 * (self.Re/r)**2 * self.J2 * np.sin(phi)*np.cos(phi)           # 地球南北周方向の重力ベクトルの大きさ．0-90deg:Gp>0, 90-180deg:Gp<0

        # 地球中心方向の重力ベクトルの各成分
        Grx = Gr * rx / r
        Gry = Gr * ry / r
        Grz = Gr * rz / r

        # 地球南北周方向の重力ベクトルの各成分
        Gpz = - Gp * r_xy / r
        Gpxy = Gp * rz / r
        Gpx = Gpxy * rx / r_xy
        Gpy = Gpxy * ry / r_xy

        # 合算
        Gx = Grx + Gpx
        Gy = Gry + Gpy
        Gz = Grz + Gpz

        Gravity = np.array([[Gx], [Gy], [Gz]])
        return Gravity

    def ENUtoECI(self, r_e, v_e, qbe):
        # ENU座標系（東経，北緯，高度の座標系）からECI座標系への変換
        # r_e : [East[rad],North[rad],Altitude[m]]
        # GRS80を用いている

        CosN = np.cos(r_e[1, 0])
        SinN = np.sin(r_e[1, 0])
        CosE = np.cos(r_e[0, 0])
        SinE = np.sin(r_e[0, 0])
        Al = r_e[2, 0]

        # 姿勢の変換行列算出(ECI⇔ENU)
        Cie0 = np.array([[CosN*CosE, CosN*SinE, SinN], [-SinE, CosE, 0], [-SinN*CosE, -SinN*SinE, CosN]])
        Cie1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        Cie = np.dot(Cie1, Cie0)
        Cei = np.linalg.inv(Cie)

        # 測地緯度算出
        Geocentric_Lat = r_e[1, 0]
        Geodetic_Lat = np.arctan(1/(1 - self.f)/(1 - self.f)*np.tan(Geocentric_Lat))    # 測地緯度
        CosNG = np.cos(Geodetic_Lat)                                                    #
        SinNG = np.sin(Geodetic_Lat)

        # ECI位置計算
        r_is = np.zeros([3, 1])     # 緯度・経度から決まる地球表面の位置ベクトル
        r_ia = np.zeros([3, 1])     # 対地高度から決まる，地球表面からその位置までの位置ベクトル
        r_i = np.zeros([3, 1])      # ECIでの位置ベクトル

        # 扁平率から長半径，短半径，その緯度での半径を算出
        ra = self.Re
        rb = self.Re*(1 - self.f)
        r_dash = np.sqrt(1 / ((CosN/ra)**2 + (SinN/rb)**2))

        r_is[0, 0] = CosE * CosN * r_dash  # 地球中心->標準時子午線がX軸
        r_is[1, 0] = SinE * CosN * r_dash  #
        r_is[2, 0] = SinN * r_dash

        r_ia[0, 0] = CosE * CosNG * Al  # 地球中心->標準時子午線がX軸
        r_ia[1, 0] = SinE * CosNG * Al  #
        r_ia[2, 0] = SinNG * Al

        r_i = r_is + r_ia

        # ECI速度計算
        v_ip = np.cross(self.OmeE.T, r_i.T).T       # その地点自体のECIでの速度
        v_iv = np.dot(Cei, v_e)                     # ENU系での速度ベクトルがECI系でどの速度ベクトルか．
        v_i = v_ip + v_iv                           # ECIでの速度ベクトル

        # ECIクォータニオン計算
        Cbe = self.Quate_ToDCM(qbe)
        Cbi = np.dot(Cei, Cbe)
        qbi = self.DCM_ToQuate(Cbi)

        return r_i, v_i, qbi