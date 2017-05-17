import numpy as np
import pygame

from lib.pipeline import capture_process, extract_process
from multiprocessing import Process, Queue
from queue import Empty

if __name__ == "__main__":

    capture_fps_queue, extraction_fps_queue = Queue(), Queue()

    capture_queue = Queue(1)
    capture = Process(target=capture_process.start, args=(capture_queue, capture_fps_queue))
    capture.start()

    extraction_out = Queue(1)
    extract = Process(target=extract_process.start, args=(capture_queue, extraction_out))
    extract.start()

    (frame, timestamp, entities) = extraction_out.get()
    capture_fps = 0.0
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]), pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE)

    pygame.font.init()
    font = pygame.font.Font(pygame.font.get_default_font(), 20)

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            while not capture_fps_queue.empty():
                capture_fps = capture_fps_queue.get()

            try:
                (frame, timestamp, entities) = extraction_out.get_nowait()
            except Empty as e:
                pass

            screen.fill((0, 0, 0))

            pygame.surfarray.blit_array(screen, np.swapaxes(frame, 0, 1))
            #pygame.draw.circle(screen, (0, 255, 0), (entities['ship'][1], entities['ship'][0]), 2)

            for blue_bullet in entities['blue_bullets']:
                pygame.draw.circle(screen, (255, 0, 0), (int(blue_bullet[1]), int(blue_bullet[0])), 2)

            capture_fps_text = font.render("Capture FPS: {}".format(capture_fps), True, (255, 255, 255))
            screen.blit(capture_fps_text, (0, 0))

            pygame.image.save(screen, 'out/{}.png'.format(timestamp))

            pygame.display.flip()
    finally:
        capture.terminate()
        extract.terminate()