import pyserial

def send_to_arduino(data:list, com="COM3"):
    port = serial.Serial(com, 9600)
    port.write(data.encode('utf-8'))