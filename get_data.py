from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

act = list(
    set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))

""" HELPER FUNCTIONS """

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def timeout(func, args = (), kwargs = {}, timeout_duration = 1, default = None):
    """From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    """
    import threading

    class InterruptableThread(threading.Thread):

        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


testfile = urllib.URLopener()

# Note: you need to create the uncropped folder first in order
# for this to work

for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("subset_actors.txt"):
        if a in line:
            filename = name + str(i) + '.' + line.split()[4].split('.')[-1]
            # A version without timeout (uncomment in case you need to
            # unsupress exceptions, which timeout() does)
            # testfile.retrieve(line.split()[4], "uncropped/"+filename)

            # timeout is used to stop downloading images which take too long to
            # download
            # result = (filename, headers)
            timeout(testfile.retrieve,
                    (line.split()[4], "uncropped/" + filename), {}, 30)
            if not os.path.isfile("uncropped/" + filename):
                continue

            # crop data and save to cropped
            try:
                x1, y1, x2, y2 = map(int, line.split()[5].split(','))
                # print x1, y1, x2, y2
                im = imread("uncropped/" + filename)
                gray_im = rgb2gray(im)
                cropped_im = im[y1:y2, x1: x2]
                resized_cropped_im = imresize(cropped_im, (32, 32))
                imsave("cropped/" + filename, resized_cropped_im)

            except Exception, e:
                print "image {0} invalid, removed".format(filename)
                print str(e)
                os.remove("uncropped/" + filename)
                os.remove("cropped/" + filename)

            print filename
            i += 1
    #         break
    # break
