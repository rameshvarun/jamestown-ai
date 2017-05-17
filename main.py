import numpy as np
import pygame

from lib.pipeline import capture_process, extract_process
from multiprocessing import Process, Queue
from queue import Empty

if __name__ == "__main__":
    capture_queue = Queue(0)
    capture = Process(target=capture_process.start, args=(capture_queue, ))
    capture.start()

    extraction_out = Queue(0)
    extract = Process(target=extract_process.start, args=(capture_queue, extraction_out))
    extract.start()

    (frame, timestamp, entities) = extraction_out.get()
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            try:
                (frame, timestamp, entities) = extraction_out.get_nowait()
            except Empty as e:
                pass

            screen.fill((0, 0, 0))

            if frame is not None:
                pygame.surfarray.blit_array(screen, np.swapaxes(frame, 0, 1))

            pygame.display.flip()
    finally:
        capture.terminate()
        extract.terminate()