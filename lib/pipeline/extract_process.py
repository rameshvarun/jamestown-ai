import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from skimage import io, feature
from scipy import misc

logger.info("Loading in image templates.")

def imread(filename):
    im = io.imread(filename)[:,:,:3]
    im = misc.imresize(im, 0.5)
    return im

blue_bullet = imread('blue-bullet.png')
pink_bullet = imread('pink-bullet.png')
red_beam = imread('red-beam.png')

def find_blue_bullets(frame):
    result = np.squeeze(feature.match_template(frame, blue_bullet))
    return feature.peak_local_max(result, threshold_abs=0.7)

def find_pink_bullets(frame):
    result = np.squeeze(feature.match_template(frame, pink_bullet))
    return feature.peak_local_max(result, threshold_abs=0.7)

def find_ship(frame):
    result = np.squeeze(feature.match_template(frame, red_beam))
    return np.unravel_index(result.argmax(), result.shape)

def start(image_queue, out_queue):
    logger.info("Starting object extraction process.")
    while True:
        (frame, frame_timestamp) = image_queue.get()

        start_time = time.time()
        blue_bullets = find_blue_bullets(frame)
        pink_bullets = find_pink_bullets(frame)
        ship = find_ship(frame)
        end_time = time.time()

        logger.info("Extraction took %d seconds...", end_time - start_time)
        out_queue.put((frame, frame_timestamp, {
            'blue_bullets': blue_bullets, 'ship': ship, 'pink_bullet': pink_bullets
        }))