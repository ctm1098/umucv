import cv2          as cv
import numpy        as np
from umucv.stream import autoStream
from collections  import deque

n = 10

d = deque(maxlen = n)

def grid(deq,n):
    xs = np.hstack(deq)
    xss = np.vstack([xs]*n)
    return xss

for key,frame in autoStream():
    rimg = cv.resize(frame, (160,140))
    d.append(rimg)
    _grid = grid(d,n)
    cv.imshow('frames',_grid)
    
cv.destroyAllWindows()