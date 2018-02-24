from util import *
from pylab import *
from scipy.misc import *
import os
import urllib
import hashlib

"""
Data format:
[first name, last name, num1, num2, url, face coords[x1, y1, x2, y2], sha256]
"""


# ========== HELPER FUNCTIONS ==========
def rgb2gray(rgb):
    """
    Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray_array = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray_array / 255.


def timeout(func, args = (), kwargs = None, timeout_duration = 1, default = None):
    """
    From:
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


def check_sha256(file_path, target_sha256):
    """
    Check the sha256 for the downloaded file
    :param file_path: the downloaded file path
    :type file_path: str
    :param target_sha256: target_sha256 of the downloaded file
    :type target_sha256: str
    :return: True if the target_sha256 is correct
    :rtype: bool
    """
    # check target_sha256
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read()
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest() == target_sha256


def print_actor_count():
    """
    Print the actor : count dictionary
    """
    for actor, count in actor_count.iteritems():
        print "{} : {}".format(actor, count)


def crop_image(actor_info, uncropped_path, cropped_path, gray = True, resolution = 32):
    """
    Crop, grayscale and resize the downloaded image, and save.
    If the image could not be processed, delete it.
    :param actor_info: the info about the actor
    :type actor_info: list
    :param uncropped_path: the uncropped (src) path
    :type uncropped_path: str
    :param cropped_path: the cropped (dest) path
    :type cropped_path: str
    :param resolution: the resolution. options: 32 or 64
    :type resolution: int
    :return: True if the image is successfully processed
    :rtype: bool
    """
    coords = actor_info[5]
    try:
        x1, y1, x2, y2 = map(int, coords.split(','))
        im = imread(uncropped_path)
        if gray:
            im = rgb2gray(im)
        cropped_im = im[y1:y2, x1: x2]
        resized_cropped_im = imresize(cropped_im, (resolution, resolution))
        imsave(cropped_path, resized_cropped_im)
        return True

    except Exception, e:
        print e
        # if os.path.isfile(uncropped_path):
        #     os.remove(uncropped_path)
        # if os.path.isfile(cropped_path):
        #     os.remove(cropped_path)
        return False


# ========== ANSWERS ==========
# NOTE: create uncropped and cropped folder before running get_data()
def get_actor_data(actor):
    """
    Download data of a given actor, compare their SHA256 and crop his/her faces
    :param actor: the acter's full name
    :type actor: str
    :return:
    :rtype:
    """
    testfile = urllib.URLopener()
    i = 0
    for line in open(ACTORS_FILE):
        actor_info = line.split()
        if len(actor_info) < 7:
            continue
        first_name = actor_info[0]
        last_name = actor_info[1]
        url = actor_info[4]
        target_sha256 = actor_info[6]

        # check if line is about target actor
        if actor.split()[0].lower() == first_name.lower() and \
                        actor.split()[1].lower() == last_name.lower():
            file_name = "{0}{1}.{2}".format(last_name.lower(), str(i), url.split('.')[-1])
            uncropped_path = UNCROPPED + file_name
            cropped32_path = CROPPED32 + file_name
            cropped64_path = CROPPED64 + file_name
            cropped227_path = "Resource/cropped_227/" + file_name

            # download file
            timeout(testfile.retrieve, (url, uncropped_path), {}, 30)

            if not os.path.isfile(uncropped_path):
                print "Failed to download {0}".format(file_name)
                print "\t{}".format(actor_info)
                continue

            if not check_sha256(uncropped_path, target_sha256):
                print "Incorrect hash for {0}".format(file_name)
                if os.path.isfile(uncropped_path):
                    os.remove(uncropped_path)
                continue

            if crop_image(actor_info, uncropped_path, cropped32_path, True, 32) and \
                    crop_image(actor_info, uncropped_path, cropped64_path, True, 64) and \
                    crop_image(actor_info, uncropped_path, cropped227_path, False, 227):
                print file_name + " OK"
                i += 1
            else:
                print "Error processing {}, pass".format(file_name)

    print "========== Done: {0} has {1} images ==========".format(actor, i)


def get_data():
    """
    Download data for all actors, compare their SHA256 and crop their faces
    """
    for actor in actors:
        get_actor_data(actor)


if __name__ == "__main__":
    get_data()
    # print_actor_count()
