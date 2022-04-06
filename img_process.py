import math
import cv2
import numpy as np
import stack

center_width = 320
center_height = 240
angle_flag = 0
circle_flag = 0

cap = cv2.VideoCapture(0)

lineDataStack = stack.Stack()


# detect circle and draw and recognize it's color
def detect_circle(img):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # dilate
    gray = cv2.dilate(gray, None, iterations=2)
    # erode
    gray = cv2.erode(gray, None, iterations=2)
    cv2.imshow("circle", gray)
    # 霍夫圆检测参数：输入圆的图像、检测模式、累加器分辨率与图片分辨率的反比、圆心检测的阈值、圆心检测的累加器阈值（数值越大圆越小），圆心检测的最小半径，圆心检测的最大半径
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=110, minRadius=20, maxRadius=150)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cv2.putText(img, "Find circle", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
        # ser.write("Find_circle".encode("utf-8"))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 3)


# 标准霍夫线变换
def line_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 二值化
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst = cv2.dilate(dst, None, iterations=2)  # 膨胀
    dst = cv2.erode(dst, None, iterations=6)  # 腐蚀
    # cv2.imshow("line_detection", dst)
    edges = cv2.Canny(dst, 50, 150, apertureSize=3)  # 边缘检测
    # cv2.imshow("edges", edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # 检测直线
    if lines is not None:
        store_line = [1, 1]
        for line in lines:
            if lines.all:
                rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
                a = np.cos(theta)  # theta是弧度
                b = np.sin(theta)
                x0 = a * rho  # 代表x = r * cos（theta）
                y0 = b * rho  # 代表y = r * sin（theta）
                # 直角检测
                if store_line[0] * x0 + store_line[1] * y0 < 1:
                    cv2.putText(frame, "Find right angle", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
                    # ser.write("Find_right_angle".encode("utf-8"))
                store_line[0] = x0
                store_line[1] = y0
                cv2.circle(frame, (int(x0 + center_width), int(y0 + center_height)), 1, (0, 0, 255), 2)
                x1 = int(x0 + 500 * (-b))  # 计算直线起点横坐标
                y1 = int(y0 + 500 * a)  # 计算起始起点纵坐标
                x2 = int(x0 - 500 * (-b))  # 计算直线终点横坐标
                y2 = int(y0 - 500 * a)  # 计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                cv2.circle(frame, (int(center_x), int(center_y)), 1, (0, 255, 0), 2)
                lineDataStack.push(str(center_width - center_x) + " " + str(center_height - center_y))
                if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 400:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


def start():
    while 1:
        ret, frame = cap.read()
        detect_circle(frame)
        line_detection(frame)
        cv2.imshow("image-lines", frame)
        # line_detect_possible_demo(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
