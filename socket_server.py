import socket
import cv2
import time

sock = socket.socket()
sock.bind(('', 9090))
sock.listen(1)
conn, addr = sock.accept()

print("connected" , addr)

while True:
    img = b""
    while True:
        st = time.time()
        data = conn.recv(2097152)
        #if not data:
            #conn.sendall(b"recieved image")
            #break
        #img += data
        conn.sendall(b"bytes recieved ")
        print(len(data), "bytes")
        print("time: ", time.time()-st)

conn.close()