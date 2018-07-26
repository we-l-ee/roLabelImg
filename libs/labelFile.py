# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from base64 import b64encode, b64decode
from pascal_voc_io import PascalVocWriter
from pascal_voc_io import XML_EXT
import os.path
import sys
import math


class LabelFileError(Exception):
    pass

def addRotatedShape(shapes, label, cx, cy, hw, hh, angle, difficult):

    p0x,p0y = rotatePoint(cx,cy, cx - hw, cy - hh, -angle)
    p1x,p1y = rotatePoint(cx,cy, cx + hw, cy - hh, -angle)
    p2x,p2y = rotatePoint(cx,cy, cx + hw, cy + hh, -angle)
    p3x,p3y = rotatePoint(cx,cy, cx - hw, cy + hh, -angle)

    points = [(p0x, p0y), (p1x, p1y), (p2x, p2y), (p3x, p3y)]
    shapes.append((label, points, angle, True, None, None, difficult))

def rotatePoint(xc,yc, xp,yp, theta):
    xoff = xp-xc;
    yoff = yp-yc;

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    # pRes = (xc + pResx, yc + pResy)
    return xc+pResx,yc+pResy


def yolo_format(class_index, pmins, pmaxs, width, height):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x_center = (pmins[0] + pmaxs[0]) / float(2.0 * width)
    y_center = (pmins[1] + pmaxs[1]) / float(2.0 * height)
    x_width = float(abs(pmaxs[0] - pmins[0])) / width
    y_height = float(abs(pmaxs[1] - pmins[1])) / height
    return str(class_index) + " " + str(x_center) \
           + " " + str(y_center) + " " + str(x_width) + " " + str(y_height)


def norm_bb_format(class_index, pmins, pmaxs, width, height):
    xmin = pmins[0] / float(width)
    ymin = pmins[1] / float(height)
    xmax = pmaxs[0] / float(width)
    ymax = pmaxs[1] / float(height)
    return str(class_index) + " " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax)


def norm_rbb_format(class_index, rbbox, width, height):
    cx = rbbox[0] / float(width)
    cy = rbbox[1] / float(height)
    w = rbbox[2] / float(width)
    h = rbbox[3] / float(height)

    return str(class_index) + " " + str(cx) + " " + str(cy) + " " + str(w) + " " + str(h) + " " + str(rbbox[4])


def save_bb(myfile, line):
    myfile.write(line + "\n")  # append line


class LabelFile(object):
    # It might be changed as window creates. By default, using XML ext
    # suffix = '.lif'
    suffix = XML_EXT

    def __init__(self, filename=None, labels=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        self.verified = False

        self.labels = labels
        self.invLabels = {v: k for k, v in self.labels.items()}

        self.lineColor = QColor(0, 0, 255)
        self.fillColor = QColor(255, 0, 0)

        print(self.labels)

    def readAll_bbox(self, path, w, h):
        bnd_boxes = []
        with open(path, 'r') as fin:
            for line in fin.readlines():
                line = line.strip()
                toks = line.split(" ")
                cx, cy = float(toks[1]) * w, float(toks[2]) * h
                hw, hh = float(toks[3]) * w / 2.0, float(toks[4]) * h / 2.0


                print(cy, cx, hw, hh)

                if len(toks) == 6:
                    addRotatedShape(bnd_boxes,self.invLabels[int(toks[0])], cx, cy, hw, hh, float(toks[5]), False)
                else:
                    assert "Rotation only"

        return bnd_boxes

    def saveAll_bbox(self, annPath, shapes, w, h):
        isRotated_check = None
        with open(str(annPath), 'w') as myfile:
            for shape in shapes:
                points = shape['points']
                label = shape['label']
                # Add Chris
                difficult = int(shape['difficult'])
                direction = shape['direction']
                isRotated = shape['isRotated']
                if isRotated_check is None:
                    isRotated_check = isRotated

                assert (isRotated_check == isRotated)
                pmins, pmaxs = LabelFile.convertPoints2BndBox2(points)
                print(pmins, pmaxs)
                # if shape is normal box, save as bounding box
                # print('direction is %lf' % direction)

                bndbox = norm_bb_format(self.labels[str(label)], pmins, pmaxs, w, h)
                if not isRotated:
                    save_bb(myfile, bndbox)
                else:  # if shape is rotated box, save as rotated bounding box
                    robndbox = LabelFile.convertPoints2RotatedBndBox(shape)

                    save_bb(myfile, norm_rbb_format(self.labels[str(label)], robndbox, w, h))

    def saveYoloFormat(self, annPath, shapes, imagePath, imageData):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        image = QImage()
        image.load(imagePath)
        # imageShape = [image.height(),,
        #               1 if image.isGrayscale() else 3]

        isRotated_check = None
        with open(str(annPath), 'w') as myfile:
            for shape in shapes:
                points = shape['points']
                label = shape['label']
                # Add Chris
                difficult = int(shape['difficult'])
                direction = shape['direction']
                isRotated = shape['isRotated']
                if isRotated_check is None:
                    isRotated_check = isRotated

                assert (isRotated_check == isRotated)
                pmins, pmaxs = LabelFile.convertPoints2BndBox2(points)
                print(pmins, pmaxs)
                # if shape is normal box, save as bounding box 
                # print('direction is %lf' % direction)
                bndbox = yolo_format(self.labels[str(label)], pmins, pmaxs, image.width(), image.height())
                if not isRotated:
                    save_bb(myfile, bndbox)
                else:  # if shape is rotated box, save as rotated bounding box
                    save_bb(myfile, bndbox + " " + str(direction))

        return

    def savePascalVocFormat(self, filename, shapes, imagePath, imageData,
                            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        image = QImage()
        image.load(imagePath)
        imageShape = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]
        writer = PascalVocWriter(imgFolderName, imgFileNameWithoutExt,
                                 imageShape, localImgPath=imagePath)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            # Add Chris
            difficult = int(shape['difficult'])
            direction = shape['direction']
            isRotated = shape['isRotated']
            # if shape is normal box, save as bounding box 
            # print('direction is %lf' % direction)
            if not isRotated:
                bndbox = LabelFile.convertPoints2BndBox(points)
                writer.addBndBox(bndbox[0], bndbox[1], bndbox[2],
                                 bndbox[3], label, difficult)
            else:  # if shape is rotated box, save as rotated bounding box
                robndbox = LabelFile.convertPoints2RotatedBndBox(shape)
                writer.addRotatedBndBox(robndbox[0], robndbox[1],
                                        robndbox[2], robndbox[3], robndbox[4], label, difficult)

        writer.save(targetFile=filename)
        return

    def toggleVerify(self):
        self.verified = not self.verified

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix

    @staticmethod
    def convertPoints2BndBox2(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin)), (int(xmax), int(ymax))

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))

    # You Hao, 2017/06/121
    @staticmethod
    def convertPoints2RotatedBndBox(shape):
        points = shape['points']
        center = shape['center']
        direction = shape['direction']

        cx = center.x()
        cy = center.y()

        w = math.sqrt((points[0][0] - points[1][0]) ** 2 +
                      (points[0][1] - points[1][1]) ** 2)

        h = math.sqrt((points[2][0] - points[1][0]) ** 2 +
                      (points[2][1] - points[1][1]) ** 2)

        angle = direction % math.pi

        return (round(cx, 4), round(cy, 4), round(w, 4), round(h, 4), round(angle, 6))
