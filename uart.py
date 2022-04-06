import serial

# 打开串口
def OpenPort(bps, timeout):
    try:
        ser = serial.Serial('/dev/ttyAMA0', bps, timeout=timeout)
        if ser.is_open == False:
            ser = -1
    except Exception as e:
        print("===异常===", e)
    return ser
