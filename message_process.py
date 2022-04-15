from struct import pack

import main

def send_message(message,type):
    sumA = 0
    sumB = 0
    data = bytearray([0x42])
    main.ser().write(data)
    data = bytearray([0x02, 8])
    for b in data:
        sumB = sumB + b
        sumA = sumA + sumB
    main.ser().write(data)
    data = message
    bytes = pack(type, data)
    for b in bytes:
        sumB = sumB + b
        sumA = sumA + sumB
    main.ser().write(bytes)
    data = bytearray([sumB, sumA])
    main.ser().write(data)