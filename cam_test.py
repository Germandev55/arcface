import cv2

#Global_vars.cap1 = cv2.VideoCapture("rtsp://10.24.72.33:554/0")
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture("rtsp://admin:Admin1234@92.125.152.58:6461")
#Global_vars.cap1 = cv2.VideoCapture("rtsp://admin:admin@10.24.72.31:554/Streaming/Channel/101")

## rtsp://192.168.2.109:554/user=admin&password=mammaloe&channel=1&stream=0.sdp?
## rtsp://89.239.192.188:553/ucast/11

#Global_vars.cap2 = cv2.VideoCapture("rtsp://viewer:viewernstu1@172.16.4.67:80")

print("cap1 init done")
cv2.namedWindow("cam1", cv2.WINDOW_NORMAL) 
cv2.namedWindow("cam2", cv2.WINDOW_NORMAL) 

while 1:
    try:
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break
        
        cap1_grab = cap1.grab()
        _, stream_buffer1 = cap1.retrieve(cap1_grab)

        cap2_grab = cap2.grab()
        _, stream_buffer2 = cap2.retrieve(cap2_grab)

        cv2.imshow("cam1", stream_buffer1)
        cv2.imshow("cam2", stream_buffer2)

    except:
        pass

cap1.release()
cap2.release()

cv2.destroyAllWindows()