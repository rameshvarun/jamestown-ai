import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from lib.capture import  WindowCapture
from lib.movingaverage import EWMA
import time

def start(out_queue, report_fps):
    logger.info("Starting image capture process.")
    capture = WindowCapture("Jamestown")
    fps = EWMA(0.9)

    while True:
        start_time = time.time()
        frame = capture.capture_as_array(downscale=4)
        end_time = time.time()

        fps.observe(1/(end_time - start_time))
        report_fps.put(fps.get())

        out_queue.put((frame, time.time()))
