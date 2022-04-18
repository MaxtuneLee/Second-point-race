import threading
import time
import imgprocess as imgp
import uart
import main
import stack

lineDataStack = stack.Stack()
distDataStack = stack.Stack()
angleDataStack = stack.Stack()


# 定义线程捏，记得*add可接收多个以非关键字方式传入的参数
def lineData():
    while True:
        # 暂停 0.5 秒后，再执行
        time.sleep(0.5)
        if distDataStack.isEmpty() is not True or angleDataStack.isEmpty() is not True:
            # print("lineDataStack is not empty")
            # data = lineDataStack.pop()
            dist = distDataStack.pop()
            angle = angleDataStack.pop()
            uart.send_distance_message(dist)
            uart.send_angle_message(angle)
            # print("[Notice] stream.py:", data)
            # uart.sendData(main.ser, data)


# def graphData():
#     while True:
#         if imgp.circle_flag == 1:
#             print("[Notice] stream.py:", "circle found")
#         if imgp.angle_flag == 1:
#             print("[Notice] stream.py:", "line found")


lineDataStream = threading.Thread(target=lineData)
# graphicStream = threading.Thread(target=graphData)
# 启动线程
lineDataStream.start()
# graphicStream.start()

# 主线程执行如下语句
print("[Notice] stream.py: ", threading.current_thread().getName() + "running")
