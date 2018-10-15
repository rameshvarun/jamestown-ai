import numpy as np
import pygame

from scipy import misc

from lib.pipeline import capture_process, extract_process
from multiprocessing import Process, Queue, Pool
from queue import Empty
from skimage import io

import click

FRAME_SCALE = 4.0

cli = click.Group()

def draw_extract_debug(screen, frame, entities):
    pygame.surfarray.blit_array(screen, np.swapaxes(frame, 0, 1))
    if entities['ship'] is not None:
        pygame.draw.circle(screen, (0, 255, 0), (int(FRAME_SCALE*entities['ship'][1]), int(FRAME_SCALE*entities['ship'][0])), 5)

    for blue_bullet in entities['blue_bullets']:
        pygame.draw.circle(screen, (255, 0, 0), (int(FRAME_SCALE*blue_bullet[1]), int(FRAME_SCALE*blue_bullet[0])), 4)
    for pink_bullet in entities['pink_bullets']:
        pygame.draw.circle(screen, (255, 0, 0), (int(FRAME_SCALE*pink_bullet[1]), int(FRAME_SCALE*pink_bullet[0])), 4)

@cli.command()
@click.argument('file', type=click.File('rb'))
def frametest(file):
    pool = Pool()
    frame = extract_process.imread(file)
    entities = extract_process.extract_elements(pool, frame)

    frame = misc.imresize(frame, FRAME_SCALE, 'nearest')
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]), pygame.HWSURFACE | pygame.DOUBLEBUF)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        draw_extract_debug(screen, frame, entities)
        pygame.display.flip()

@cli.command()
def run():
    capture_fps_queue, extraction_fps_queue = Queue(), Queue()

    capture_queue = Queue(1)
    capture = Process(target=capture_process.start, args=(capture_queue, capture_fps_queue))
    capture.start()

    extraction_out = Queue(1)
    extract = Process(target=extract_process.start, args=(capture_queue, extraction_out, extraction_fps_queue))
    extract.start()

    (frame, timestamp, entities) = extraction_out.get()
    frame = misc.imresize(frame, FRAME_SCALE, 'nearest')

    capture_fps, extract_fps = 0.0, 0.0
    screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]), pygame.HWSURFACE | pygame.DOUBLEBUF)

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
            while not extraction_fps_queue.empty():
                extract_fps = extraction_fps_queue.get()

            try:
                (frame, timestamp, entities) = extraction_out.get_nowait()
                frame = misc.imresize(frame, FRAME_SCALE, 'nearest')
            except Empty as e:
                pass

            screen.fill((0, 0, 0))
            draw_extract_debug(screen, frame, entities)

            capture_fps_text = font.render("Capture FPS: {:.2f}".format(capture_fps), True, (255, 255, 255))
            screen.blit(capture_fps_text, (0, 0))

            extract_fps_text = font.render("Extract FPS: {:.2f}".format(extract_fps), True, (255, 255, 255))
            screen.blit(extract_fps_text, (0, 25))

            pygame.display.flip()
    finally:
        capture.terminate()
        extract.terminate()

if __name__ == "__main__":
    cli()
