import math
from struct import pack

import cv2
import numpy as np
import stack
import stream
import uart
import main

center_width = 320
center_height = 240
angle_flag = 0
circle_flag = 0
color_flag = "null"
font = cv2.FONT_HERSHEY_SIMPLEX
lower_red = np.array([0, 150, 150])
higher_red = np.array([10, 255, 255])
lower_green = np.array([35, 110, 106])  # 绿色阈值下界
higher_green = np.array([77, 255, 255])  # 绿色阈值上界

cap = cv2.VideoCapture(2)

lineDataStack = stack.Stack()
lineDataStack.push(0)

# ser = uart.OpenPort()



# 蠢方法：阈值化找直线计算黑色区域像素中心点和镜头中心点的距离和偏移方向（太不优雅了）
def findLine(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    dst = cv2.dilate(dst, None, iterations=2)
    dst = cv2.erode(dst, None, iterations=6)
    cv2.imshow("find line center", dst)
    color = dst[400]
    black_num = np.sum(color == 0)
    black_index = np.where(color == 0)
    if black_num == 0:
        black_num = 1
    if black_index[0].size != 0:
        center = (black_index[0][black_num - 1] + black_index[0][0]) >> 1
        direction = center - 320
        stream.lineDataStack.push(direction)


# detect circle and draw and recognize it's color
def detect_circle(img):
    circle_flag = 0
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
        circle_flag = 1
        # ser.write("Find_circle".encode("utf-8"))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 3)


# 标准霍夫线变换
def line_detection(frame):
    angle_flag = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 二值化
    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dst = cv2.dilate(dst, None, iterations=2)  # 膨胀
    dst = cv2.erode(dst, None, iterations=6)  # 腐蚀
    # cv2.imshow("line_detection", dst)
    edges = cv2.Canny(dst, 50, 150, apertureSize=3)  # 边缘检测
    cv2.imshow("edges", edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)  # 检测直线
    if lines is not None:
        avg_center_x = 0
        avg_center_y = 0
        store_line = [1, 1]
        store_line2 = [1, 1]
        store_line3 = [1, 1]
        for line in lines:
            if lines.all:
                rho, theta = line[0]
                # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                a = np.cos(theta)  # theta是弧度
                b = np.sin(theta)
                x0 = a * rho  # 代表x = r * cos（theta）
                y0 = b * rho  # 代表y = r * sin（theta）
                # 直角检测
                if store_line[0] * x0 + store_line[1] * y0 < 1:
                    cv2.putText(frame, "Find right angle", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
                    angle_flag = 1
                    # ser.write(bytearray("angle"))
                    print("Find right angle")
                    # ser.write("Find_right_angle".encode("utf-8"))
                cv2.circle(frame, (int(x0 + center_width), int(y0 + center_height)), 1, (0, 0, 255), 2)
                x1 = int(x0 + 100 * (-b))  # 计算直线起点横坐标
                y1 = int(y0 + 100 * a)  # 计算起始起点纵坐标
                x2 = int(x0 - 100 * (-b))  # 计算直线终点横坐标
                y2 = int(y0 - 100 * a)  # 计算直线终点纵坐标
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                if x0 * store_line[1] - y0 * store_line[0] < 1:
                    if (math.sqrt(math.pow((x1 - store_line2[0]), 2) + math.pow((y1 - store_line2[1]), 2))) > 10:
                        avg_center_x = (x0 + store_line[0]) / 2
                        avg_center_y = (y0 + store_line[1]) / 2
                rho2 = avg_center_x / a
                rho3 = avg_center_y / b
                rho_avg = (rho2 + rho3) / 2
                stream.distDataStack.push(rho_avg*0.1)
                stream.angleDataStack.push(theta)
                # uart.send_angle_message(theta)
                # uart.send_distance_message(rho_avg*0.1)

                # # 按照acfly的循线串口协议处理发送数据
                # if theta > 90:
                #     theta_error = theta - 180
                # else:
                #     theta_error = theta
                # output_str = "%f" % theta_error
                # sumA = 0
                # sumB = 0
                # data = bytearray([0x41, 0x43])
                # ser.write(data)
                # data = bytearray([0x02, 8])
                # for b in data:
                #     sumB = sumB + b
                #     sumA = sumA + sumB
                # ser.write(data)
                # float_value = theta_error
                # float_bytes = pack('f', float_value)
                # for b in float_bytes:
                #     sumB = sumB + b
                #     sumA = sumA + sumB
                # ser.write(float_bytes)
                # float_value = rho_avg * 0.1
                # float_bytes = pack('f', float_value)
                # for b in float_bytes:
                #     sumB = sumB + b
                #     sumA = sumA + sumB
                # ser.write(float_bytes)
                # data = bytearray([sumB, sumA])
                # ser.write(data)

                x3 = int(avg_center_x + 500 * (-b))  # 计算直线起点横坐标
                y3 = int(avg_center_y + 500 * a)  # 计算起始起点纵坐标
                x4 = int(avg_center_x - 500 * (-b))
                y4 = int(avg_center_y - 500 * a)
                cv2.circle(frame, (int(avg_center_x), int(avg_center_y)), 1, (0, 255, 0), 2)
                # stream.lineDataStack.push(str(center_width - center_x) + " " + str(center_height - center_y))
                if math.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2) > 400:
                    cv2.line(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > 400:
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                store_line3[0] = center_x
                store_line3[1] = center_y
                store_line2[0] = x1
                store_line2[1] = y1
                store_line[0] = x0
                store_line[1] = y0
    # else:
    #     sumA = 0
    #     sumB = 0
    #     data = bytearray([0x41, 0x43])
    #     ser.write(data)
    #
    #     data = bytearray([0x02, 8])
    #     for b in data:
    #         sumB = sumB + b
    #         sumA = sumA + sumB
    #     ser.write(data)
    #
    #     float_value = 200
    #     float_bytes = pack('f', float_value)
    #     for b in float_bytes:
    #         sumB = sumB + b
    #         sumA = sumA + sumB
    #     ser.write(float_bytes)
    #
    #     float_value = 0
    #     float_bytes = pack('f', float_value)
    #     for b in float_bytes:
    #         sumB = sumB + b
    #         sumA = sumA + sumB
    #     ser.write(float_bytes)
    #
    #     data = bytearray([sumB, sumA])
    #     ser.write(data)


def recognizeColor(frame):
    color_flag = "null"
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(img_hsv, lower_red, higher_red)  # 可以认为是过滤出红色部分，获得红色的掩膜,去掉背景
    mask_red = cv2.medianBlur(mask_red, 7)  # 中值滤波(把数字图像中的一点的值用该点的邻域各点的中值代替，让 周围像素值接近真实值，从而消除孤立的噪声点)
    mask_green = cv2.inRange(img_hsv, lower_green, higher_green)  # 获得绿色部分掩膜
    mask_green = cv2.medianBlur(mask_green, 7)  # 中值滤波
    # mask_black = cv2.inRange(img_hsv, lower_black, higher_black)  # 获得绿色部分掩膜
    # mask_black = cv2.medianBlur(mask_black, 7)  # 中值滤波

    # mask = cv2.bitwise_or(mask_red, mask_red)  # 三部分掩膜进行按位或运算

    cnts1, hierarchy1 = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓检测
    # cnts2, hierarchy2 = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts3, hierarchy3 = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if cnts1 is not None:
        color_flag = 'red'

    for cnt in cnts1:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(frame, (x, y - 20), (x + w, y + h), (0, 0, 255), 2)  # 将检测到的颜色框起来
        cv2.putText(frame, 'red', (x, y - 20), font, 0.7, (0, 0, 255), 2)

    # for cnt in cnts2:
    #     (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)  # 将检测到的颜色框起来
    #     cv2.putText(frame, 'black', (x, y - 1), font, 0.7, (0, 0, 0), 2)

    if cnts3 is not None:
        color_flag = 'green'

    for cnt in cnts3:
        (x, y, w, h) = cv2.boundingRect(cnt)  # 该函数返回矩阵四个点
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h), (0, 255, 0), 2)  # 将检测到的颜色框起来
        cv2.putText(frame, 'green', (x, y - 50), font, 0.7, (0, 255, 0), 2)


def recognizeCircle(img):
    recognizeColor(img)
    detect_circle(img)
    if circle_flag == 1 and color_flag == "green":
        print("laser")
        # ser.write(bytearray("check"))
    if circle_flag == 1 and color_flag == "red":
        print("start point")
        # ser.write(bytearray("start"))


def start():
    while 1:
        ret, frame = cap.read()
        recognizeCircle(frame)
        findLine(frame)
        line_detection(frame)
        cv2.imshow("image-lines", frame)
        # line_detect_possible_demo(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
