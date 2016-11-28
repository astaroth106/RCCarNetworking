import cv2
import io
import numpy as np
import socket
import struct
from PIL import Image
import RyanZotti as stop
# Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
# all interfaces)
server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')
try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        bytes = ' '
        bytes += connection.read(image_len)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        if a != -1 and b != -1:
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            print "whoa images?"
            stop.detect_stop_sign(i)
            cv2.imshow('i', i)
            if cv2.waitKey(1) == 27:
                exit(0)
            # Rewind the stream, open it as an image with PIL and do some
            # processing on it
        #image_stream.seek(0)
        #image = Image.open(image_stream)
        #cv2.imshow('image', image)

finally:
    connection.close()
    server_socket.close()
