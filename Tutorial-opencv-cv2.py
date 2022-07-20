import cv2
import numpy as np
from matplotlib import pyplot as plt
import urllib.request
import os
import re


def one_scriptImages_readShowWrite():
    # CV2 read image
    img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
    # CV2 show image, close with any key: waitKey(0)
    cv2.imshow('image: one_scriptImages_readShowWrite()',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # matplotlib show image
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
    plt.show()
    # CV2 write image
    cv2.imwrite('results/watch_gray.png',img)


def two_scriptVideo_capture():
    # choose webcam 0,1 or 2, etc.
    cap = cv2.VideoCapture(0)
    #
    while (True):
        # read video feed
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray
        # show feed in window: quit with q
        cv2.imshow('webcam: two_scriptVideo_capture()', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #
    cap.release()
    cv2.destroyAllWindows()


def two_scriptVideo_captureRecord():
    # video capture: choose webcam 0,1 or 2, etc.
    cap = cv2.VideoCapture(0)
    # video writer:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('results/output.avi', fourcc, 20.0, (640, 480))
    #
    while (True):
        # read video feed
        ret, frame = cap.read()
        # convert video feet to: gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray
        # record video feed: in color
        out.write(frame)
        # show gray feed in window: quit with q
        cv2.imshow('webcam: two_scriptVideo_captureRecord()', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def two_scriptVideo_loadPlay():
    # video capture: choose webcam 0,1 or 2, etc.
    cap = cv2.VideoCapture('results/output.avi')
    #
    while (True):
        # read video feed
        ret, frame = cap.read()
        # convert video feet to: gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray
        # show gray feed in window: quit with q
        cv2.imshow('webcam: two_scriptVideo_captureRecord()', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #
    cap.release()
    cv2.destroyAllWindows()


def three_scriptImage_resizeDrawingWritingOnIt():
    # CV2 create: resizable window with name
    cv2.namedWindow('image: three_scriptImage_resizeDrawingWritingOnIt()', cv2.WINDOW_NORMAL)
    # CV2 read image: in color
    img = cv2.imread('Screenshot.png', cv2.IMREAD_COLOR)
    # scale image: dsize
    scale_percent = 80
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(img, dsize)

    # draw line, where: on the image, where_start, where_end, color: blue-green-red, width: px
    cv2.line(img, (0, 0), (200, 300), (255, 255, 255), 50)
    # draw rectangle, where: on the image, top_left, bottom_right, color, width
    cv2.rectangle(img, (500, 250), (1000, 500), (0, 0, 255), 15)
    # draw circle, where: on the image, center, radius, color, width: px or -1 to fill
    cv2.circle(img, (447, 63), 63, (0, 255, 0), -1)
    # numpy array of: some points, datatype;
    # draw polyline, where: on the image, connect point, connect first and last point: true, false, color, width
    pts = np.array([[100, 50], [200, 300], [700, 200], [500, 100]], np.int32)
    pts = pts.reshape((-1, 1, 2)) #reshape probably not necessary
    cv2.polylines(img, [pts], True, (0, 255, 255), 3)
    # write text, where: on the image, what text, where start, font, size , color, width,
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV Tuts!', (10, 500), font, 12, (200, 255, 155), 33, cv2.LINE_AA)
    # CV2 show image: in Named Window; close with any key: waitKey(0)
    cv2.imshow('image: three_scriptImage_resizeDrawingWritingOnIt()', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def four_scriptImage_operations():
    # CV2 create: resizable window with name
    cv2.namedWindow('image: four_scriptImage_operations()', cv2.WINDOW_NORMAL)
    # CV2 read image: in color
    img = cv2.imread('Screenshot.png', cv2.IMREAD_COLOR)
    # scale image: dsize
    scale_percent = 80
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    size = (width, height)
    img = cv2.resize(img, size)

    # do analysis on image, position: px; print color: blue-green-red
    px = img[55, 55]
    print(px)
    # change color of px:
    img[55, 55] = [255, 255, 255]
    print(px)
    # do analysis on range/region of image, ROI: px
    px = img[100:150, 100:150]
    print(px)
    # change color of px: in x von 200-550, in y von 300-450
    img[200:550, 300:450] = [255, 255, 255]
    # shape: (height, length, ); size: number of pixels; dtype: data type
    print(img.shape)
    print(img.size)
    print(img.dtype)
    # image region: cut out and paste
    watch_face = img[600:800,300:600]
    img[0:200, 0:300] = watch_face
    #
    cv2.imshow('image: four_scriptImage_operations()', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def five_scriptImage_addition():
    # CV2 create: resizable window with name
    cv2.namedWindow('image: five_scriptImage_addition()', cv2.WINDOW_NORMAL)
    # CV2 read image: in color
    img1 = cv2.imread('3D-Matplotlib.png')
    img2 = cv2.imread('mainsvmimage.png')
    # scale image: dsize
    scale_percent = 200
    width1 = int(img1.shape[1] * scale_percent / 100)
    height1 = int(img1.shape[0] * scale_percent / 100)
    width2 = int(img2.shape[1] * scale_percent / 100)
    height2 = int(img2.shape[0] * scale_percent / 100)
    size1 = (width1, height1)
    size2 = (width2, height2)
    img1 = cv2.resize(img1, size1)
    img2 = cv2.resize(img2, size2)
    cv2.imshow('image: five_scriptImage_addition()', img1)
    cv2.waitKey(0)
    cv2.imshow('image: five_scriptImage_addition()', img2)
    cv2.waitKey(0)
    #
    add = img1 + img2
    cv2.imshow('image: five_scriptImage_addition()', add)
    cv2.waitKey(0)
    # too bright, because: (155,211,79) + (50, 170, 200) = 205, 381, 279...translated to (205, 255,255).
    add = cv2.add(img1, img2)
    cv2.imshow('image: five_scriptImage_addition()', add)
    cv2.waitKey(0)
    # image1, weight1, image2, weight2, gamma: some measure of light
    weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
    cv2.imshow('image: five_scriptImage_addition()', weighted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def five_ScriptImage_additionMask():
    # CV2 create: resizable window with name
    cv2.namedWindow('image: five_ScriptImage_additionMask()', cv2.WINDOW_NORMAL)
    # CV2 read image: in color
    img1 = cv2.imread('3D-Matplotlib.png')
    img2 = cv2.imread('mainlogo.png')
    # scale image: dsize
    scale_percent = 200
    width1 = int(img1.shape[1] * scale_percent / 100)
    height1 = int(img1.shape[0] * scale_percent / 100)
    width2 = int(img2.shape[1] * scale_percent / 100)
    height2 = int(img2.shape[0] * scale_percent / 100)
    size1 = (width1, height1)
    size2 = (width2, height2)
    img1 = cv2.resize(img1, size1)
    img2 = cv2.resize(img2, size2)
    cv2.imshow('image: five_ScriptImage_additionMask()', img1)
    cv2.waitKey(0)
    cv2.imshow('image: five_ScriptImage_additionMask()', img2)
    cv2.waitKey(0)

    # I want to put logo on top-left corner, So I create a ROI
    # row-px:252, column-px: 252, color-channels: 3 (no alpha)
    rows, cols, channels = img2.shape
    print(rows, cols, channels)
    img1_roi = img1[0:rows, 0:cols]
    cv2.imshow('image: five_ScriptImage_additionMask()', img1_roi)
    cv2.waitKey(0)

    # convert color from bgr to gray
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image: five_ScriptImage_additionMask()', img2gray)
    cv2.waitKey(0)

    # create a mask of logo, by adding a threshold: where to apply,
    # anything below 220 will be 0, anything over 220 will be 255,
    # THRESH_BINARY_INV: flip / invert, anything below 220 will be 255, anything over 220 will be 0
    ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('image: five_ScriptImage_additionMask()', mask)
    cv2.waitKey(0)

    # create inverse mask
    mask_inv = cv2.bitwise_not(mask)
    cv2.imshow('image: five_ScriptImage_additionMask()', mask_inv)
    cv2.waitKey(0)

    # black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(img1_roi, img1_roi, mask=mask_inv)
    cv2.imshow('image: five_ScriptImage_additionMask()', img1_bg)
    cv2.waitKey(0)

    # take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    cv2.imshow('image: five_ScriptImage_additionMask()', img2_fg)
    cv2.waitKey(0)

    # add
    dst = cv2.add(img1_bg, img2_fg)
    # paste
    img1[0:rows, 0:cols] = dst
    cv2.imshow('image: five_ScriptImage_additionMask()', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def six_ScriptImage_thresholding():
    # CV2 create: resizable window with name
    cv2.namedWindow('image: six_ScriptImage_thresholding()', cv2.WINDOW_NORMAL)
    # CV2 read image: in color
    img = cv2.imread('bookpage.jpg', cv2.IMREAD_COLOR)
    # scale image: dsize
    scale_percent = 80
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(img, dsize)

    # gray image
    img_grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresholds for low light image: everything with a small amount of color will enhance respective channel to 255
    retval, threshold_1 = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
    # thresholds for low light image: everything with a small amount of light will become white / 255
    retval, threshold_2 = cv2.threshold(img_grayscaled, 10, 255, cv2.THRESH_BINARY)
    # thresholds for low light image: gaussian adaptive
    threshold_3 = cv2.adaptiveThreshold(img_grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow('image: six_ScriptImage_thresholding()', img)
    cv2.waitKey(0)
    cv2.imshow('image: six_ScriptImage_thresholding()', threshold_1)
    cv2.waitKey(0)
    cv2.imshow('image: six_ScriptImage_thresholding()', threshold_2)
    cv2.waitKey(0)
    cv2.imshow('image: six_ScriptImage_thresholding()', threshold_3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def seven_scriptVideo_filterForRed():
    # choose webcam 0,1 or 2, etc.
    cap = cv2.VideoCapture(0)
    #
    while (1):
        # read video feed
        _, frame = cap.read()
        # convert color: bgr to hsv hue=color, saturation=grayness/intensity, value=light
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # numpy arrays for: three color values in HSV (not RGB or blue-green-red)
        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])
        # The way this works is what we see, will be anything, that is between our ranges here:
        # basically 30-255, 150-255, and 50-180.
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        #
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        # ESC-key: top-left
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #
    cv2.destroyAllWindows()
    cap.release()


def eight_scriptVideo_blurringSmoothing():
    # choose webcam 0,1 or 2, etc.
    cap = cv2.VideoCapture(0)
    #
    while (1):
        # read video feed
        _, frame = cap.read()
        # convert color: bgr to hsv hue=color, saturation=grayness/intensity, value=light
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # numpy arrays for: three color values in HSV (not RGB or blue-green-red)
        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])
        # The way this works is what we see, will be anything, that is between our ranges here:
        # basically 30-255, 150-255, and 50-180.
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # average of 15 by 15 pixels
        kernel = np.ones((15, 15), np.float32) / 225
        smoothed = cv2.filter2D(res, -1, kernel)
        #
        blur = cv2.GaussianBlur(res, (15, 15), 0)
        # least noisy one
        median = cv2.medianBlur(res, 15)
        #
        bilateral = cv2.bilateralFilter(res, 15, 75, 75)
        #
        cv2.imshow('frame', frame)
        cv2.imshow('smoothed', smoothed)
        cv2.imshow('blur', blur)
        cv2.imshow('median', median)
        cv2.imshow('bilateral', bilateral)
        # ESC-key: top-left
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        #
    cv2.destroyAllWindows()
    cap.release()


def nine_scriptVideo_morphologicalTransformation():
    # choose webcam 0,1 or 2, etc.
    cap = cv2.VideoCapture(0)
    #
    while (1):
        # read video feed
        _, frame = cap.read()
        # convert color: bgr to hsv hue=color, saturation=grayness/intensity, value=light
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # numpy arrays for: three color values in HSV (not RGB or blue-green-red)
        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])
        # The way this works is what we see, will be anything, that is between our ranges here:
        # basically 30-255, 150-255, and 50-180.
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # 5x5 Matrix filled with 1
        kernel = np.ones((5, 5), np.uint8)
        # erosion: object eroded away; dilation: object dilated, but also background noise
        erosion = cv2.erode(mask, kernel, iterations=1)
        dilation = cv2.dilate(mask, kernel, iterations=1)
        # removes false positives (background noise); closing removes false negatives (black pixels in object)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', mask)
        cv2.imshow('erosion', erosion)
        cv2.imshow('dilation', dilation)
        cv2.imshow('opening', opening)
        cv2.imshow('closing', closing)

        # ESC-key: top-left
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        #
    cv2.destroyAllWindows()
    cap.release()


def ten_scriptVideo_edgeDetectionGradients():
    # choose webcam 0,1 or 2, etc.
    cap = cv2.VideoCapture(0)
    #
    while (1):
        # read video feed
        _, frame = cap.read()
        # convert color: bgr to hsv hue=color, saturation=grayness/intensity, value=light
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # numpy arrays for: three color values in HSV (not RGB or blue-green-red)
        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])
        # The way this works is what we see, will be anything, that is between our ranges here:
        # basically 30-255, 150-255, and 50-180.
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # detects edges: which image; higher numbers: less edge detection
        edges = cv2.Canny(frame, 100, 200)
        # detects gradients: which image, what data type
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        # detects gradients: which image, what data type, x, y, kernel size
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
        #
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('Edges', edges)
        cv2.imshow('laplacian', laplacian)
        cv2.imshow('sobelx', sobelx)
        cv2.imshow('sobely', sobely)
        # ESC-key: top-left
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        #
    cv2.destroyAllWindows()
    cap.release()


def eleven_scriptImage_templateMatching():
    # CV2 create: resizable window with name
    cv2.namedWindow('image: eleven_scriptImage_templateMatching()', cv2.WINDOW_NORMAL)
    # CV2 read image: in color
    img1 = cv2.imread('opencv-template-matching-python-tutorial.jpg')
    img2 = cv2.imread('opencv-template-for-matching.jpg', 0)
    # scale image: dsize
    scale_percent = 200
    width1 = int(img1.shape[1] * scale_percent / 100)
    height1 = int(img1.shape[0] * scale_percent / 100)
    width2 = int(img2.shape[1] * scale_percent / 100)
    height2 = int(img2.shape[0] * scale_percent / 100)
    size1 = (width1, height1)
    size2 = (width2, height2)
    img1 = cv2.resize(img1, size1)
    img2 = cv2.resize(img2, size2)
    cv2.imshow('image: eleven_scriptImage_templateMatching()', img1)
    cv2.waitKey(0)
    cv2.imshow('image: eleven_scriptImage_templateMatching()', img2)
    cv2.waitKey(0)
    # image: gray
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # just a way to get: width and height
    w, h = img2.shape[::-1]
    print(w, h)
    # looking for the matches: of img2 in img1_gray
    res = cv2.matchTemplate(img1_gray, img2, cv2.TM_CCOEFF_NORMED)
    # res higher than threshold: in percent.
    # The lower the threshold, the more matches, but also more false positives
    # Better way than lowering threshold: use more templates
    threshold = 0.75
    loc = np.where(res >= threshold)
    print(loc)
    # mark these locations
    # *: seems to be an iterator; [::-1]: seems to switch elements, so pt gets correct x,y coordinates
    # draw rectangle, where: on the image, top_left, bottom_right, color, width
    print(loc[::-1])
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img1, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        print(pt)
    #
    cv2.imshow('image: eleven_scriptImage_templateMatching()', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def twelve_scriptImage_grabCutForegroundExtraction():
    # CV2 create: resizable window with name
    cv2.namedWindow('image: twelve_scriptImage_grabCutForegroundExtraction()', cv2.WINDOW_NORMAL)
    # CV2 read image: in color
    img = cv2.imread('opencv-python-foreground-extraction-tutorial.jpg')
    # scale image: dsize
    scale_percent = 100
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(img, dsize)
    # mask: black image of same size as img
    # img.shape[:2]: px in y-direction (height), px in x-direction (width)
    mask = np.zeros(img.shape[:2], np.uint8)
    print(img.shape[:2])
    # Model: array of zeros, 1 row, 65 columns
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    print(bgdModel)
    # rect: search area; x, y, x-length, y-length
    rect = (161, 79, 150, 150)
    # grab and cut a certain region: no need to understand the details of three lines
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_res = img * mask2[:, :, np.newaxis]
    #
    cv2.imshow('image: twelve_scriptImage_grabCutForegroundExtraction()', img)
    cv2.waitKey(0)
    cv2.imshow('image: twelve_scriptImage_grabCutForegroundExtraction()', mask)
    cv2.waitKey(0)
    cv2.imshow('image: twelve_scriptImage_grabCutForegroundExtraction()', img_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #
    plt.imshow(img_res)
    plt.colorbar()
    plt.show()


def thirteen_scriptImage_cornerDetection():
    # CV2 create: resizable window with name
    cv2.namedWindow('image: thirteen_scriptImage_cornerDetection()', cv2.WINDOW_NORMAL)
    # CV2 read image: in color
    img = cv2.imread('opencv-corner-detection-sample.jpg', cv2.IMREAD_COLOR)
    # scale image: dsize
    scale_percent = 80
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(img, dsize)
    #
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)
    # find where: in img_gray, find 100 at most, "quality" number < 1 and the lower the mor detection, minimum distance between corners in px
    corners = cv2.goodFeaturesToTrack(img_gray, 200, 0.01, 10)
    corners = np.int0(corners)
    # mark: with circle
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    #
    cv2.imshow('image: thirteen_scriptImage_cornerDetection()', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fourteen_scriptImage_featureMatching():
    # CV2 create: resizable window with name
    cv2.namedWindow('image: fourteen_scriptImage_featureMatching()', cv2.WINDOW_NORMAL)
    # CV2 read image: 0 in gray (default color)
    img1 = cv2.imread('opencv-feature-matching-template.jpg', 0)
    img2 = cv2.imread('opencv-feature-matching-image.jpg', 0)
    # scale image: dsize
    scale_percent = 100
    width1 = int(img1.shape[1] * scale_percent / 100)
    height1 = int(img1.shape[0] * scale_percent / 100)
    width2 = int(img2.shape[1] * scale_percent / 100)
    height2 = int(img2.shape[0] * scale_percent / 100)
    size1 = (width1, height1)
    size2 = (width2, height2)
    img1 = cv2.resize(img1, size1)
    img2 = cv2.resize(img2, size2)
    cv2.imshow('image: fourteen_scriptImage_featureMatching()', img1)
    cv2.waitKey(0)
    cv2.imshow('image: fourteen_scriptImage_featureMatching()', img2)
    cv2.waitKey(0)
    # orb: detector of similarity
    orb = cv2.ORB_create()
    # key points, descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # find matching: key points and descriptors;
    # sort them based on: distance (confidence), key=lambda x: x.distance -> key(x) returns x.distance
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # resulting image: image1, kp1, image2, kp2, number of matches shown,
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    #
    cv2.imshow('image: fourteen_scriptImage_featureMatching()', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #
    plt.imshow(img3)
    plt.show()


def fiveteen_scriptVideo_backgroundReduction():
    # detects motion in video: says motion is foreground, no motion is background.
    # tree leaves are moving as well
    #
    # choose webcam 0,1 or 2, etc. / or a prerecorded video
    cap = cv2.VideoCapture('people-walking.mp4')
    # motion detector, background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
    #
    while (1):
        # read video feed
        ret, frame = cap.read()
        # apply background subtractor to: frame
        fgmask = fgbg.apply(frame)
        #
        cv2.imshow('frame', frame)
        cv2.imshow('fgmask', fgmask)
        # ESC-key: top-left
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    #
    cap.release()
    cv2.destroyAllWindows()


def sixteen_scriptVideo_haarCascadeObjectDetectionFaceEye():
    # Download:
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    # Download Raw: click Raw > RHC & save file as (otherwise: CascadeClassifier complains!)
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    #
    # load classifiers
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # choose webcam 0,1 or 2, etc. / or a prerecorded video
    cap = cv2.VideoCapture(0)
    #
    while 1:
        # read video feed
        ret, frame = cap.read()
        # make feed: gray
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in: frame_gray
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        #
        for (x, y, w, h) in faces:
            # show rectangle: in frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # define roi ranges
            roi_gray = frame_gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            # detect eyes in: roi_gray
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # show rectangle: in roi_color
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        #
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    #
    cap.release()
    cv2.destroyAllWindows()


def seventeen_scriptImage_haarCascadeCreation_downloadImages():
    # make new directory neg: in path of this script
    if not os.path.exists('neg'):
        os.makedirs('neg')
    # URL to download lots of images: change hardcoded url in function below
    seventeen_ScriptImage_get_imagesURLsAndImages_from_websiteUnsplash()


def seventeen_ScriptImage_get_imagesURLsAndImages_from_websiteUnsplash():
    # add other url here
    url = 'https://unsplash.com/s/photos/wristwatch'
    # url request returns: binary response text
    resp = urllib.request.urlopen(url).read()
    # url pattern to search for in binary response text
    urls_1 = re.findall(b'https://media.istockphoto.com/photos/.*?="', resp)
    urls_2 = re.findall(b'https://media.istockphoto.com/photos/.*?=\\\\"', resp)
    # the last one is useless
    urls_1 = urls_1[:-1]
    # add urls into one list
    urls = []
    # with watermark
    for url_ in urls_1:
        url = url_[:-1]
        #urls.append(url)
    # without watermark
    for url_ in urls_2:
        url = url_[:-2]
        urls.append(url)
    #
    # download the images
    i = 0
    for url in urls:
        # decode binary to text utf-8
        url = url.decode('utf-8')
        # path to store images
        i = i + 1
        #directory = 'C:/Users/41792/Downloads/'
        directory = 'neg/'
        filename = url.split('/')[-1]
        filename = str(i) + filename[:20] + '.png'
        path = os.path.join(directory, filename)
        print(path, url)
        # request download: from url and store in path
        urllib.request.urlretrieve(url, path)
        #
        # load as: gray
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # resize: 100x100 px
        resized_image = cv2.resize(img, (100, 100))
        # save image, overwrite downloaded image
        cv2.imwrite(path, resized_image)


def seventeen_scriptImage_createPosAndNeg():
    # switch: run for 'neg' and run for 'pos'
    # we won't run 'pos' because we only want to detect the one watch, not others
    # we apply opencv_createsamples.exe to create samples of the one watch
    for file_type in ['neg']:
        print(file_type)
        # iterate through: all images in folder 'neg' or 'pos'
        for img in os.listdir(file_type):
            print(img)
            #
            if file_type == 'pos':
                # write line into file info.dat
                line = file_type + '/' + img + ' 1 0 0 50 50\n'
                with open('info.dat', 'a') as f:
                    f.write(line)
            elif file_type == 'neg':
                # write line into file bg.txt
                line = file_type + '/' + img + '\n'
                with open('bg.txt', 'a') as f:
                    f.write(line)


def seventeen_scriptImage_haarCascadeCreation_training():
    pass
    # this approach will be disabled: modern approaches via DNN - tensorflow deep neural networks - provides much better results
    #
    # create folders: opencv_workspace, info, data
    # install: opencv for windows
    # copy/paste into opencv_workspace: neg, watch5050.jpg, bg.txt
    #
    #opencv_workspace
    #--neg
    #----negimages.jpg
    #--opencv
    #--info
    #--data
    #--positives.vec --bg.txt
    #--watch5050.jpg
    #
    # the rest is done on the command line with:
    # opencv_createsamples -img watch5050.jpg -bg bg.txt -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1950
    # opencv_createsamples -info info/info.lst -num 1950 -w 20 -h 20 -vec positives.vec
    # opencv_traincascade -data data -vec positives.vec -bg bg.txt -numPos 1800 -numNeg 900 -numStages 10 -w 20 -h 20


def seventeen_scriptImage_haarCascadeCreation_usage():
    # load classifiers
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # this is the cascade we just made.
    watch_cascade = cv2.CascadeClassifier('watchcascade10stage.xml')
    # choose webcam 0,1 or 2, etc. / or a prerecorded video
    cap = cv2.VideoCapture(0)
    #
    while 1:
        # read video feed
        ret, frame = cap.read()
        # make feed: gray
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in: frame_gray
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        # detect watches in: frame_gray; reject levels level weights.
        watches = watch_cascade.detectMultiScale(frame_gray, 50, 50)
        for (x, y, w, h) in watches:
            # show rectangle: in frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            # write text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Watch', (x - w, y - h), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)
        #
        for (x, y, w, h) in faces:
            # show rectangle: in frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # define roi ranges
            roi_gray = frame_gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            # detect eyes in: roi_gray
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # show rectangle: in roi_color
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        #
        cv2.imshow('frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    #
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # choose scripts to run
    # ----------------

    if 0:
        one_scriptImages_readShowWrite()
        two_scriptVideo_capture()
        two_scriptVideo_captureRecord()
        two_scriptVideo_loadPlay()
        three_scriptImage_resizeDrawingWritingOnIt()
        four_scriptImage_operations()
        five_scriptImage_addition()
        five_ScriptImage_additionMask()
        six_ScriptImage_thresholding()
        seven_scriptVideo_filterForRed()
        eight_scriptVideo_blurringSmoothing()
        nine_scriptVideo_morphologicalTransformation()
        ten_scriptVideo_edgeDetectionGradients()
        eleven_scriptImage_templateMatching()
        twelve_scriptImage_grabCutForegroundExtraction()
        thirteen_scriptImage_cornerDetection()
        fourteen_scriptImage_featureMatching()
        fiveteen_scriptVideo_backgroundReduction()
        sixteen_scriptVideo_haarCascadeObjectDetectionFaceEye()
        seventeen_scriptImage_haarCascadeCreation_downloadImages()
    else:
        pass


