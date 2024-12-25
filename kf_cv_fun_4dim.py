import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
from numpy.random import randn
from numpy import dot
from scipy.linalg import inv


class PosSensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]

    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        return [self.pos[0] + randn() * self.noise_std,
                self.pos[1] + randn() * self.noise_std]


N = 30  # number of iterations
dt = 1.0  # time step
R_std = 0.35
Q_std = 0.04

M_TO_FT = 1 / 0.3048

sensor = PosSensor((0, 0), (2, .5), noise_std=R_std)
zs = np.array([sensor.read() for _ in range(N)])

tracker = KalmanFilter(dim_x=4, dim_z=2)

tracker.F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

tracker.H = np.array([[M_TO_FT, 0, 0, 0],
                      [0, M_TO_FT, 0, 0]])

tracker.R = np.eye(2) * R_std ** 2
q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std ** 2)
tracker.Q[0, 0] = q[0, 0]
tracker.Q[1, 1] = q[0, 0]
tracker.Q[2, 2] = q[1, 1]
tracker.Q[3, 3] = q[1, 1]
tracker.Q[0, 2] = q[0, 1]
tracker.Q[2, 0] = q[0, 1]
tracker.Q[1, 3] = q[0, 1]
tracker.Q[3, 1] = q[0, 1]

tracker.x = np.array([[0, 0, 0, 0]]).T
# tracker.P = np.eye(4) * 500.
tracker.P = np.eye(4) * 5.

#
x = tracker.x
P = tracker.P
F = tracker.F
H = tracker.H
R = tracker.R
Q = tracker.Q

xs, ys = [], []
for z in zs:
    tracker.predict()
    tracker.update(z)
    xs.append(tracker.x[0])
    ys.append(tracker.x[1])
plt.plot(xs, ys, 'b', label='data')

#
xs_new, ys_new = [], []
for z in zs:
    z = z.reshape(2, 1)

    # predict
    x = dot(F, x)
    P = dot(F, P).dot(F.T) + Q

    # update
    S = dot(H, P).dot(H.T) + R
    K = dot(P, H.T).dot(inv(S))
    y = z - dot(H, x)
    x += dot(K, y)
    P = P - dot(K, H).dot(P)
    xs_new.append(x[0])
    ys_new.append(x[1])

plt.plot(xs_new, ys_new, 'r', label='new')
plt.grid()
plt.legend()
plt.show()
