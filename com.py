import serial


def send_to_arduino(data: str, com="COM3"):
    port = serial.Serial(port=com, baudrate=9600, timeout=1)
    port.write(data.encode('utf-8'))
    read = port.read()
    port.close()
    return read


read = send_to_arduino(data="TEST:", com="/dev/cu.usbmodem11101")
print(read)
