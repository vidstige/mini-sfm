"""OpenGL compliant numpy matrix construction"""
import math
import numpy as np


def normalized(a, axis=-1, order=2):
    n = np.atleast_1d(np.linalg.norm(a, order, axis))
    return np.divide(a, n, axis=axis)


def ortho(l, r, b, t, n, f):
    dx = r - l
    dy = t - b
    dz = f - n
    rx = -(r + l) / (r - l)
    ry = -(t + b) / (t - b)
    rz = -(f + n) / (f - n)
    return np.matrix([[2.0/dx,0,0,rx],
                      [0,2.0/dy,0,ry],
                      [0,0,-2.0/dz,rz],
                      [0,0,0,1]])


def perspective(fovy, aspect, n, f):
    s = 1.0 / math.tan(math.radians(fovy) / 2.0)
    sx, sy = s / aspect, s
    zz = (f + n) / (n - f)
    zw = 2 * f * n / (n - f)
    return np.array([[sx,  0,  0,  0],
                      [ 0, sy,  0,  0],
                      [ 0,  0, zz, zw],
                      [ 0,  0, -1,  0]]).T


def frustum(x0, x1, y0, y1, z0, z1):
    a = (x1+x0)/(x1-x0)
    b = (y1+y0)/(y1-y0)
    c = -(z1+z0)/(z1-z0)
    d = -2*z1*z0/(z1-z0)
    sx = 2*z0/(x1-x0)
    sy = 2*z0/(y1-y0)
    return np.matrix([[sx, 0, a, 0],
                      [ 0,sy, b, 0],
                      [ 0, 0, c, d],
                      [ 0, 0,-1, 0]])
 

def translate(xyz):
    x, y, z = xyz
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])


def scale(xyz):
    x, y, z = xyz
    return np.matrix([[x,0,0,0],
                      [0,y,0,0],
                      [0,0,z,0],
                      [0,0,0,1]])

 
def rotate(a, xyz):
    x, y, z = normalized(xyz)
    s, c = math.sin(a), math.cos(a)
    nc = 1 - c
    return np.matrix([[x*x*nc +   c, x*y*nc - z*s, x*z*nc + y*s, 0],
                      [y*x*nc + z*s, y*y*nc +   c, y*z*nc - x*s, 0],
                      [x*z*nc - y*s, y*z*nc + x*s, z*z*nc +   c, 0],
                      [           0,            0,            0, 1]])
 
def rotx(a):
    s, c = math.sin(a), math.cos(a)
    return np.matrix([[1,0,0,0],
                      [0,c,-s,0],
                      [0,s,c,0],
                      [0,0,0,1]])
 
def roty(a):
    s, c = math.sin(a), math.cos(a)
    return np.matrix([[c,0,s,0],
                      [0,1,0,0],
                      [-s,0,c,0],
                      [0,0,0,1]])
 
def rotz(a):
    s, c = sincos(a)
    return np.matrix([[c,-s,0,0],
                      [s,c,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])

def lookat(eye, target, up):
    F = target[:3] - eye[:3]
    f = normalized(F)
    U = normalized(up[:3])
    s = np.cross(f, U)
    u = np.cross(s, f)
    M = np.identity(4)
    #import sys
    #print(s, u, f, file=sys.stderr)
    M[:3, :3] = np.vstack([s, u, -f])
    T = translate(-eye)
    return np.dot(M, T)


def viewport(x, y, w, h):
    x, y, w, h = map(float, (x, y, w, h))
    return np.matrix([[w/2, 0  , 0,x+w/2],
                      [0  , h/2, 0,y+h/2],
                      [0  , 0  , 1,    0],
                      [0  , 0  , 0,    1]])

