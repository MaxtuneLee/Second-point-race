import threading
import time
import imgprocess as imgp
import uart
import main
import stack

lineDataStack = stack.Stack()

# 定义线程捏，记得*add可接收多个以非关键字方式传入的参数
def lineData():
    while True:
        # 暂停 0.3 秒后，再执行
        time.sleep(0.5)
        if lineDataStack.isEmpty() is not True:
            data = lineDataStack.pop()
            print("[Notice] stream.py:", data)
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
