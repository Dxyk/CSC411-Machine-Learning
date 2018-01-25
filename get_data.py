from faces import *
from pylab import *
from scipy.misc import *
import os
import urllib

# list of actors
actors = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))


# HELPER FUNCTIONS
def rgb2gray(rgb):
    """Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray_array = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray_array / 255.


def timeout(func, args = (), kwargs = None, timeout_duration = 1,
            default = None):
    """From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/
    """
    import threading

    # noinspection PyTypeChecker,PyCallByClass,PyBroadException
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


# NOTE: create uncropped and cropped folder before running get_data()
def get_data():
    testfile = urllib.URLopener()
    for actor in actors:
        actor_name = actor.split()[1].lower()
        i = 0
        for line in open("subset_actors.txt"):
            if actor in line:
                # generate file name and download image
                filename = actor_name + str(i) + '.' + \
                           line.split()[4].split('.')[-1]
                timeout(testfile.retrieve,
                        (line.split()[4], "uncropped/" + filename), {}, 30)
                if not os.path.isfile("uncropped/" + filename):
                    continue

                # crop, grayscale and resize downloaded image, and save to cropped
                try:
                    x1, y1, x2, y2 = map(int, line.split()[5].split(','))
                    im = imread("uncropped/" + filename)
                    gray_im = rgb2gray(im)
                    cropped_im = gray_im[y1:y2, x1: x2]
                    resized_cropped_im = imresize(cropped_im, (32, 32))
                    imsave("cropped/" + filename, resized_cropped_im)
                    print filename + " OK"
                    i += 1

                except Exception, e1:
                    print "Error image {0}: {1}".format(filename, str(e1))
                    print "\t{0}".format(line)
                    if os.path.isfile("uncropped/" + filename):
                        os.remove("uncropped/" + filename)
                    if os.path.isfile("cropped/" + filename):
                        os.remove("cropped/" + filename)

        print "Done: {0} has {1} images".format(actor_name, i)


if __name__ == "__main__":
    # get_data()
    count_actors()
    for actor, count in actor_count.iteritems():
        print "{} : {}".format(actor, count)
