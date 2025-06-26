import os, cv2, queue, threading

# 1️⃣  Tell FFmpeg to pull the RTSP stream over TCP instead of UDP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# 2️⃣  Optional: raise the socket buffer so bursts of packets aren’t dropped
#     (values are in frames; pick whatever your RAM and latency budget allow)
BUFFERSIZE = 60             # hold up to 60 decoded frames
CAP_OPTS   = cv2.CAP_FFMPEG  # we’ll pass this constant each time

url = "rtsp://admin:admin@567@192.168.1.38:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(url, CAP_OPTS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFERSIZE)

# 3️⃣  Read continuously in a background thread so your CPU-heavy code
#     never blocks the network socket.
frames = queue.Queue(maxsize=BUFFERSIZE)

def reader():
    while True:
        ok, frame = cap.read()          # returns False if FFmpeg gives up
        if not ok:
            print("Lost stream – check network or camera")
            break
        try:
            frames.put_nowait(frame)    # drop if the queue is full
        except queue.Full:
            pass                        # we’re behind; skip this frame

threading.Thread(target=reader, daemon=True).start()

# 4️⃣  Your main thread can now process at its own pace
while True:
    frame = frames.get()                # waits for the next available frame
    # ---- heavy CV/AI work here ----
    cv2.imshow("RTSP", frame)
    if cv2.waitKey(1) == 27:            # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
