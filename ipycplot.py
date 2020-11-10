import numpy as np
import numgl
from ipycanvas import hold_canvas, RoughCanvas


def homogenize(euclidian, w=1):
    return np.vstack((euclidian, w * np.ones((1, euclidian.shape[1]))))


def dehomogenize(homogenous):
    return homogenous[:-1, :] / homogenous[-1, :]


def transform(transform, points):
    """Transforms the points, by first homogenizing them and dehomogenize them after"""
    return dehomogenize(transform @ homogenize(points.T)).T
    
def stroke_line(canvas, p1, p2):
    canvas.stroke_line(p1[0], p1[1], p2[0], p2[1])

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    tmp = iter(iterable)
    return zip(tmp, tmp)

def stroke_lines(canvas, mvp, lines, scale=None, translation=None):
    """Lines is a 2n x 3 matrix with start and end points interleaved"""
    # project with camera
    transformed_lines = scale * dehomogenize(mvp @ homogenize(lines.T)).T + translation

    with hold_canvas(canvas):
        for p1, p2 in pairwise(transformed_lines):
            stroke_line(canvas, p1, p2)


def rectangle(width, height, center=np.array([0, 0, 0])):
    """Create lines for a rectangle"""
    r = np.array([
        [0, 0, 0], [1, 0, 0],
        [1, 0, 0], [1, 1, 0],
        [1, 1, 0], [0, 1, 0],
        [0, 1, 0], [0, 0, 0],
    ])
    return (r - center) * np.array([width, height, 1])


class Camera:
    def __init__(self, canvas):
        self.target = np.array([0, 0, 0])
        self.up = np.array([0, 1, 0])
        self.eye = None
        self.projection = numgl.perspective(90, canvas.width/canvas.height, 0.1, 5)

    def lookat(self, eye=None, target=None, up=None):
        if eye is not None:
            self.eye = eye
        if target is not None:
            self.target = target
        if up is not None:
            self.up = up

    def pose(self):
        return numgl.lookat(self.eye, self.target, self.up)
        
    def mvp(self):
        return self.projection @ self.pose()

    
class Plot3D:
    def __init__(self, canvas=None):
        self.canvas = canvas or RoughCanvas()
        self.camera = Camera(self.canvas)
        self.scale = np.array([1, 1, 1])
        self.translation = np.array([self.canvas.width/2, self.canvas.height/2, 0])
    
    def rects(self, points, size=10):
        transformed = transform(self.camera.mvp(), points) * self.scale + self.translation
        x = transformed[:, 0] 
        y = transformed[:, 1]
        self.canvas.fill_rects(x, y, size)
    
    def circles(self, points, size=10):
        transformed = transform(self.camera.mvp(), points) * self.scale + self.translation
        x = transformed[:, 0] 
        y = transformed[:, 1]
        self.canvas.stroke_circles(x, y, size)
        
    def lines(self, lines):
        stroke_lines(self.canvas, self.camera.mvp(), lines, self.scale, self.translation)

    def show(self):
        return self.canvas

    
class Plot2D:
    def __init__(self, canvas=None):
        self.canvas = canvas or RoughCanvas()
        self.scale = np.array([1, 1])
        self.translation = np.array([0, 0])

    def lines(self, lines):
        with hold_canvas(self.canvas):
            for p1, p2 in pairwise(lines * self.scale + self.translation):
                self.canvas.stroke_line(
                    p1[0], p1[1],
                    p2[0], p2[1])

    def image(self, im):
        self.canvas.put_image_data(im, 0, 0)

    def show(self):
        return self.canvas

    
# plot = Plot(RoughCanvas())
# plot.lines(*rectangle(320, 200), R=R, t=t))
# plot.lines(pixels, pixels + inv(K) @ pixels)
# plot.camera(R=R, t=t, (320, 200))
# plot.show()