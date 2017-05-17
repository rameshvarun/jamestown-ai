import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from lib.capture import  WindowCapture

import time

def start(out_queue):
    logger.info("Starting image capture process.")
    capture = WindowCapture("Jamestown")

    while True:
        frame = capture.capture_as_array(downscale=2)
        out_queue.put((frame, time.time()))
