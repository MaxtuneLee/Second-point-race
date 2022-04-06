import threading
import time
import img_process as imgp


# 定义线程要调用的方法，*add可接收多个以非关键字方式传入的参数
def lineData():
    while True:
        # 暂停 1 秒后，再执行
        time.sleep(1)
        if imgp.lineDataStack.isEmpty() is not True:
            data = imgp.lineDataStack.pop()
            print("[Notice] stream.py:", data)


# def graphData():
#     while True:
#         # 暂停 1 秒后，再执行
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
