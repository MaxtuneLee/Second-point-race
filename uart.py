from struct import pack

import serial

# 打开串口
import main


def OpenPort(bps, timeout):
    try:
        ser = serial.Serial('/dev/ttyAMA0', bps, timeout=timeout)
        if ser.is_open == False:
            ser = -1
        else:
            ser.write(
                ",---.    ,---.   ____     _____     __ ,---------.   ___    _ ,---.   .--.    .-''-.   \n|    \  /    | .'  __ `.  \   _\   /  /\          \.'   |  | ||    \  |  |  .'_ _   \  \n|  ,  \/  ,  |/   '  \  \ .-./ ). /  '  `--.  ,---'|   .'  | ||  ,  \ |  | / ( ` )   ' \n|  |\_   /|  ||___|  /  | \ '_ .') .'      |   \   .'  '_  | ||  |\_ \|  |. (_ o _)  | \n|  _( )_/ |  |   _.-`   |(_ (_) _) '       :_ _:   '   ( \.-.||  _( )_\  ||  (_,_)___| \n| (_ o _) |  |.'   _    |  /    \   \      (_I_)   ' (`. _` /|| (_ o _)  |'  \   .---. \n|  (_,_)  |  ||  _( )_  |  `-'`-'    \    (_(=)_)  | (_ (_) _)|  (_,_)\  | \  `-'    / \n|  |      |  |\ (_ o _) / /  /   \    \    (_I_)    \ /  . \ /|  |    |  |  \       /  \n'--'      '--' '.(_,_).' '--'     '----'   '---'     ``-'`-'' '--'    '--'   `'-..-'   \n\n===Raspberry pi is ready===\n[notice] Pi: Type 'exit' to close uart.\n\n".encode(
                    "utf-8"))
        return ser
    except Exception as e:
        print("===异常===\n", e)
        return -1


ser = OpenPort(115200, 1)


def sendData(ser, data):
    try:
        ser.write(data.encode("utf-8"))
    except Exception as e:
        print("===数据发送异常===\n", e)


def send_angle_message(angle: int):
    data = bytearray([0x42])
    print("[Notice] uart.py:" + str(angle))
    if ser != -1:
        ser.write(data)
        ser.write(angle)


def send_distance_message(distance: int):
    data = bytearray([0x44])
    print("[Notice] uart.py:" + str(distance))
    if ser != -1:
        ser.write(data)
        ser.write(distance)
