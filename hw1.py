import cv2
import tensorflow as tf
from tensorflow.nn import conv2d
import numpy as np
from math import log, ceil, sqrt
from matplotlib import pyplot as plt

def Writer(filename='output.mp4', file_format=('m','p','4','v'), fps=30.0, videos=9, width=480, height=270):
    width *= 3
    height *= 3
    fourcc = cv2.VideoWriter_fourcc(*file_format)
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    return writer

def resizeFrame(frame,rate=0.25):
    height, width, layers = frame.shape
    new_h = int(height*rate)
    new_w = int(width*rate)
    origin = cv2.resize(frame, (new_w, new_h))
    return origin

def getGrayLevelMapping(frame):
    tempFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    c = 255 / log(1+255) # base e
    temp = np.array([ [np.uint8(c * log(1 + tempFrame[i][j])) for j in range(tempFrame.shape[1])] for i in range(tempFrame.shape[0])])
    return cv2.cvtColor(tempFrame, cv2.COLOR_GRAY2BGR), cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

def getReverse(frame):
    def reverseColor(x):
        return np.uint8((x - 128)*(-1) + 127)
    tempFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp = np.array([ [reverseColor(tempFrame[i][j]) for j in range(tempFrame.shape[1])] for i in range(tempFrame.shape[0])])
    return cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

def getEqualizeHist(frame):
    return cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)

def getPowerLaw(frame, gamma=1.5):
    tempFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    c = 255 / (255**gamma)
    temp = np.array([ [np.uint8(c * (tempFrame[i][j]**gamma)) for j in range(tempFrame.shape[1])] for i in range(tempFrame.shape[0])])
    return cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

def getByFilter(frame):
    def lowFilter(tempFrame):
        kernel_map = np.array([
        [ [[1/9]], [[1/9]], [[1/9]] ],
        [ [[1/9]], [[1/9]], [[1/9]] ],
        [ [[1/9]], [[1/9]], [[1/9]] ] ])
        kernel_map = tf.constant(kernel_map, dtype=tf.float32)
        return cv2.cvtColor(np.uint8(np.reshape(conv2d(tempFrame, kernel_map, strides=[1, 1, 1, 1],padding='SAME').numpy(),(270,480))), cv2.COLOR_GRAY2BGR)

    def gaussianFilter(tempFrame):
        kernel_map = np.array([
        [ [[1/16]], [[2/16]], [[1/16]] ],
        [ [[2/16]], [[4/16]], [[2/16]] ],
        [ [[1/16]], [[2/16]], [[1/16]] ] ])
        kernel_map = tf.constant(kernel_map, dtype=tf.float32)
        return cv2.cvtColor(np.uint8(np.reshape(conv2d(tempFrame, kernel_map, strides=[1, 1, 1, 1],padding='SAME').numpy(),(270,480))), cv2.COLOR_GRAY2BGR)

    def highFilter(tempFrame):
        kernel_map = np.array([
        [ [[-1/9]], [[-1/9]], [[-1/9]] ],
        [ [[-1/9]], [[8/9]], [[-1/9]] ],
        [ [[-1/9]], [[-1/9]], [[-1/9]] ] ])
        kernel_map = tf.constant(kernel_map, dtype=tf.float32)
        return cv2.cvtColor(np.uint8(np.reshape(conv2d(tempFrame, kernel_map, strides=[1, 1, 1, 1],padding='SAME').numpy(),(270,480))), cv2.COLOR_GRAY2BGR)

    tempFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tempFrame = np.expand_dims(np.expand_dims(tempFrame,0),-1)
    return lowFilter(tempFrame), gaussianFilter(tempFrame), highFilter(tempFrame)

def add_description(rls):
    descriptions = ['Origin', 'GrayScale', 'Gray Level Mapping', 'Reverse', 'Power Law', 'Equalize Histogram', 'Low Filter', 'Gaussian Filter', 'High Filter']
    for i in range(len(descriptions)):
        cv2.putText(rls, descriptions[i], ((i%3)*480, int(i/3)*270+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)

if __name__ == '__main__':
    cap = cv2.VideoCapture('homework_1_test_video.mp4')
    writer = Writer()
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\nCan't receive frame or stream end. Exiting ...")
            break
        origin = resizeFrame(frame)
        if origin is not None:
            grayRls,  grayLMRls = getGrayLevelMapping(origin)
            negativeRls = getReverse(origin)
            powerLawRls = getPowerLaw(origin)
            eqaRls = getEqualizeHist(origin)
            low, gaussian, high = getByFilter(origin)
            rls = np.vstack((np.hstack((origin,grayRls,grayLMRls)),np.hstack((negativeRls,powerLawRls,eqaRls)),np.hstack((low,gaussian,high))))
            add_description(rls)
            writer.write(rls)
            # cv2.imshow('ImageOperation', rls) # show just 1 frame
            # break
            count += 1
            print('\rProcessed {}/{} frames.'.format(count,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),end='')
        if cv2.waitKey(1) == 27: # esc
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
