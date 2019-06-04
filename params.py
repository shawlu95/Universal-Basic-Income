import numpy as np
from scipy.optimize import fsolve

timeStep = 0.01
totalStep = 10000

H0 = 5800
M0 = 19390
G0 = 638

UBI = 252 * 12 * timeStep
gamma = np.log(100) / 100
c = 0.826
w = 0.35
lgA = 0.001
lgB = 7528
lgC = 1500
fMultiplier = 2

econGrowth = 0.0

tI = 0.135 / 4
tC = 0
tVAT = 0.1 / 16
S = 60

def wageDecay(t, w0):
    return 0.1 + (w0 - 0.1) * np.exp(-0.05 * t)

def logistic(H):
    return lgC / (1 + np.exp(- lgA * (H - lgB)))

def solveH(M, t):
    f = lambda H: UBI * np.exp(gamma * t) - c * H - logistic(H) + (1 - tI) * wageDecay(t, w) * M

    zGuess = 6000
    z = fsolve(f, zGuess)
    return z[0]

def solveM(H, t):
    f = lambda M: (1 - tC) * c * H + fMultiplier * logistic(H) - wageDecay(t, w) * M - tVAT * M + econGrowth * M

    zGuess = 6000
    z = fsolve(f, zGuess)
    return z[0]