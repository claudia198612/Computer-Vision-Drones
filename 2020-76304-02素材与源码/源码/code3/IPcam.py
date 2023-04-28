import cv2
import imutils

# CGI IPcamare
url = 'http://192.168.1.1:80/snapshot.cgi?user=admin&pwd='
# im.src = "videostream.cgi?stream="+Status.sever_push_stream_number+"&id="+d.id;
# url = 'http://192.169.1.1:80/
# url = 'http://192.168.1.1:80/videostream.cgi?user=&pwd=&resolution=32&rate=0'
# url = 'http://192.168.1.1:80/livestream.cgi?user=admin&pwd='
cnt = 0
while True:
    timer = cv2.getTickCount()
    cap = cv2.VideoCapture(0)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if cap.isOpened():
        cnt += 1
        width, height = cap.get(3), cap.get(4)
        print(cnt, '[', width, height, ']')
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        # frame = cv2.flip(frame, -180)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow('frame', frame)
    else:
        print("Error")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
