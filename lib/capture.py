"""
This module wraps functions for capturing a window into a PIL Image object.
It uses WinAPI functions (accessed through ctypes) to get the bounds of a Window
and PIL's ImageGrab extension to get the data.
"""

from PIL import ImageGrab
from ctypes import *
import numpy as np

user32, gdi32 = windll.user32, windll.gdi32
user32.SetProcessDPIAware()

class RECT(Structure):
    _fields_ = [('left', c_long), ('top', c_long),
                ('right', c_long), ('bottom', c_long)]

    def __str__(self):
        return "({}-{}, {}-{})".format(self.left, self.right, self.top, self.bottom)


class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

    def __str__(self):
        return "({}, {})".format(self.x, self.y)

    def clear(self):
        self.x, self.y = 0, 0


class WindowCapture:
    def __init__(self, title):
        self.hwnd = user32.FindWindowW(None, title)
        if self.hwnd == 0:
            raise Exception(
                "Window with title \"{}\" not found.".format(title))
        self.topleft = POINT()
        self.clientrect = RECT()

    def capture(self):
        self.topleft.clear()
        user32.GetClientRect(self.hwnd, byref(self.clientrect))
        user32.ClientToScreen(self.hwnd, byref(self.topleft))

        return ImageGrab.grab((self.topleft.x, self.topleft.y,
                               self.topleft.x + self.clientrect.right,
                               self.topleft.y + self.clientrect.bottom))

        return ImageGrab.grab()

    def capture_as_array(self, downscale=1):
        im = self.capture()
        if downscale != 1:
            im = im.resize((im.width // downscale, im.height // downscale))

        im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((im.height, im.width, 3))
        return im_arr