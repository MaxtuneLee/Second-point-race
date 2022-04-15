import imgprocess as imgp
import stream
import uart

if __name__ == "__main__":
    print(
        "\n  ____  _                     _\n |  _ \| |                   | |\n | |_) | | ___  ___ ___    __| |_ __ ___  _ __   ___\n |  _ <| |/ _ \/ __/ __|  / _` | '__/ _ \| '_ \ / _ \\\n | |_) | |  __/\__ \__ \ | (_| | | | (_) | | | |  __/_\n |____/|_|\___||___/___/  \__,_|_|  \___/|_| |_|\___(_)\n")
    print("===程序正在運行===")
    # 启动图像处理
    imgp.start()
    uart.send_message("A","c",1)
    # 打开端口准备发送数据
    ser = uart.OpenPort()
