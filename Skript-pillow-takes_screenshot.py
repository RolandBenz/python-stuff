from PIL import ImageGrab

"""Take Screenshot"""
def pillow_takesScreenshot():
    # grab Image
    # define box position and size
    # upper left point = (y from screen left, x from screen top)
    # lower right point = (y to right from screen left, x down from screen top)
    bbox = (300, 100, 1300, 900)
    im = ImageGrab.grab(bbox)
    # save screenshot in project folder
    sFile = 'screenshot.png'
    im.save(sFile)
    im.close()


if __name__ == '__main__':
    pillow_takesScreenshot()