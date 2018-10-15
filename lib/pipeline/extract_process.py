import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import cv2

import numpy as np
from skimage import io, feature
from scipy import misc
from lib.movingaverage import EWMA

from multiprocessing import Pool

logger.info("Loading in image templates.")

def imread(filename):
    im = io.imread(filename)[:,:,:3]
    im = misc.imresize(im, 0.25)
    return im

blue_bullet = imread('patterns/blue-bullet.png')
blue_bullet_offset = np.asarray(blue_bullet.shape)[:2]/2

pink_bullet = imread('patterns/pink-bullet.png')
pink_bullet_offset = np.asarray(pink_bullet.shape)[:2]/2

red_beam = imread('patterns/red-beam.png')
red_beam_offset = np.asarray(red_beam.shape)[:2]/2

blue_beam = imread('patterns/blue-beam.png')
blue_beam_offset = np.asarray(blue_beam.shape)[:2]/2

def find_blue_bullets(frame):
    result = cv2.matchTemplate(frame, blue_bullet, cv2.TM_SQDIFF)
    return feature.peak_local_max(-result, threshold_abs=-500000) + blue_bullet_offset

def find_pink_bullets(frame):
    result = cv2.matchTemplate(frame, pink_bullet, cv2.TM_SQDIFF)
    return feature.peak_local_max(-result, threshold_abs=-500000) + blue_bullet_offset

def find_ship(frame):
    result = cv2.matchTemplate(frame, blue_beam, cv2.TM_SQDIFF)
    if result.min() > 8000000.0:
        return None

    return np.unravel_index(result.argmin(), result.shape) + red_beam_offset

def extract_elements(pool, frame):
    results = [
        pool.apply_async(find_blue_bullets, (frame,)),
        pool.apply_async(find_pink_bullets, (frame,)),
        pool.apply_async(find_ship, (frame,))]

    blue_bullets = results[0].get()
    pink_bullets = results[1].get()
    ship = results[2].get()

    return {
        'ship': ship,
        'blue_bullets': blue_bullets,
        'pink_bullets': pink_bullets
    }

def start(image_queue, out_queue, report_fps):
    logger.info("Starting object extraction process.")
    fps = EWMA(0.9)
    pool = Pool()

    while True:
        (frame, frame_timestamp) = image_queue.get()

        start_time = time.time()
        result = extract_elements(pool, frame)
        end_time = time.time()

        fps.observe(1/(end_time - start_time))
        report_fps.put(fps.get())

        out_queue.put((frame, frame_timestamp, result))
