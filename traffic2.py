import cv2 
import urllib 
import time 
import numpy as np 
#from LoadSymbols import Symbol 
#from websocket import create_connection 


#define class for References Images
class Symbol:
    def __init__(self):
        self.img = 0
        self.name = 0

# Width & Height From Camera 
width = 640 
height = 480 
 
# Difference Variables 
minDiff = 1000 
minSquareArea = 2000 
match = -1 
 
# Middle X 
MidX = width/2 
MidY = height/2 
 
# Font Type 
font = cv2.FONT_HERSHEY_SIMPLEX 
 
# Needed Variables for holding points for perspective correction 
rect = np.zeros((4, 2), dtype = "float32") 
maxW = width/2 
maxH = height/2 
dst = np.array([ 
        [0, 0], 
        [maxW - 1, 0], 
        [maxW - 1, maxH - 1], 
        [0, maxH - 1]], dtype = "float32") 
 
# Instance for Streaming 
#stream = urllib.urlopen('http://192.168.137.188:8080/?action=stream') 
#webSocket = create_connection("ws://192.168.137.188:8000")  
# Reference Images Display name & Original Name 
Reference_Symbols = ["arrowl_final.jpg","arrowr_final.jpg","arrowt_final.jpg","go_final.jpg","arrowstop_final.jpg"] 
 
Symbol_Titles = ["Turn Left 90", 
                 "Turn Right 90", 
                 "Turn Around", 
                 "Start..", 
                 "Stop!"] 
 
Actions = ["Left", "Right", "Back", "Ball", "Go", "Stop"] 

# Define Class Instances for Loading Reference Images (6 objects for 6 different images/symbols) 
symbol= [Symbol() for i in range(5)] 
def load_symbols():     
    for count in range(5):         
        image = cv2.imread(Reference_Symbols[count], cv2.COLOR_BGR2GRAY)
        symbol[count].img = cv2.resize(image,(width,height/2), interpolation = cv2.INTER_AREA)
        symbol[count].name = Symbol_Titles[count]
        print "Loading: ", symbol[count].name 
    print "All Reference Images Are Successfully Loaded!" 

def order_points(pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype = "float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

def generate_windows(): 
    # Windows to display frames 
    cv2.namedWindow("Main Frame", cv2.WINDOW_AUTOSIZE) 
    cv2.namedWindow("Matching Operation", cv2.WINDOW_AUTOSIZE) 
    cv2.namedWindow("Corrected Perspective", cv2.WINDOW_AUTOSIZE) 

def get_canny_edge(image, sigma=0.33): 
    # compute the median of the single channel pixel intensities     
    v = np.median(image) 
 
    # apply automatic Canny edge detection using the computed median     l
    lower = int(max(0, (1.0 - sigma) * v))     
    upper = int(min(255, (1.0 + sigma) * v))     
    edged = cv2.Canny(image, lower, upper)  
    # return the edged image     
    return edged 

def correct_perspective(frame, pts): 
        # Sort the contour points in Top Left, Top Right, Bottom Left & Bottom Right Manner         
        s = pts.sum(axis = 1)         
        rect[0] = pts[np.argmin(s)]         
        rect[2] = pts[np.argmax(s)]         
        diff = np.diff(pts, axis = 1)         
        rect[1] = pts[np.argmin(diff)]         
        rect[3] = pts[np.argmax(diff)] 
 
        # Compute the perspective transform matrix and then apply it         
        M = cv2.getPerspectiveTransform(rect, dst) 
        warped = cv2.warpPerspective(frame, M, (width/2,height/2))         
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) 
 
        # Calculate the maximum pixel and minimum pixel value & compute threshold 
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(warped)         
        threshold = (min_val + max_val)/2 
 
        # Threshold the image 
        ret, warped = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY) 
 
        # Return the warped image         
        return warped 

def main(): 
    load_symbols()     
    generate_windows()     
    bytes = ''     
    match = -1     
    w_save = 0     
    motor_speed = 150     
    displayText = []     
    turn = False     
    sign_found = False 

    video = cv2.VideoCapture(0)

    # capture frames from the camera
    while True:
        ret, OriginalFrame = video.read()

        # camera_frame = cv2.flip(camera_frame, 0) 
        # camera_frame = cv2.flip(camera_frame, 1) 

        # Changing color-space to grayscale & Blurring the frames to reduce the noise 
        gray = cv2.cvtColor(OriginalFrame, cv2.COLOR_BGR2GRAY)             
        blurred = cv2.GaussianBlur(gray,(3,3),0)  

        # Detecting Edges             
        edges = get_canny_edge(blurred) 

        # Contour Detection & checking for squares based on the square area 
        contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

        # Sorting contours; Taking the largest, neglecting others             
        contours = sorted(contours, key=cv2.contourArea, reverse=True) [:1] 

        for cnt in contours:                     
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True) 
            if len(approx) == 4: 
                area = cv2.contourArea(approx) 

                if area > minSquareArea: 
                    cv2.drawContours(OriginalFrame,[approx],0,(0,0,255),2)

                    warped = correct_perspective(OriginalFrame, approx.reshape(4, 2))
                    
                    for i in range(5): 
                        diffImg = cv2.bitwise_xor(warped, symbol[i].img) 
                        diff = np.count_nonzero(diffImg) 
                        if diff < minDiff:                                             
                            match = i                                             
                            #diff = minDiff                                             
                            cnt = approx.reshape(4,2)                                             
                            displayText = tuple(cnt[0])                                             
                            sign_found = True                                             
                            break

        if sign_found == True: 

            sign_found = False                 
            x,y,w,h = cv2.boundingRect(cnt)                 
            centroid_x = x + (w/2)                 
            centroid_y = y + (h/2) 

            # Real width * focal length = 16175.288                 
            camera_range = round((16175.288 / w),0) 

            # Draw the contours around sign & bounding box around sign & put the sign title 
            cv2.drawContours(OriginalFrame, [cnt], 0, (255, 0, 0), 2)                 
            cv2.rectangle(OriginalFrame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            #cv2.putText(OriginalFrame, symbol[match].name, displayText, font, 1, (255,0,255), 2, cv2.LINE_AA) 
            if w < 420: 
                if camera_range == 0 or camera_range >= 60:                        
                    turn = False                     
                else:                         
                    if w*h >= 70000 and turn == False:                             
                        if Actions[match] == "Back": 
#Back 
                            match = -1                                 
                            print "Turn Back"                                 
                            turn = True 

                        elif Actions[match] == "Left":   
#Left 
                            match = -1                                 
                            print "Left Turn"                                 
                            turn = True                             
                        elif Actions[match] == "Right": 
#Right 
                            match = -1                                 
                            print "Right Turn"                                 
                            turn = True                             
                        elif Actions[match] == "Stop":   
#Stop 
                            match = -1                                 
                            print "Stop"                                 
                            turn = True                                 
                            exit(0)             
        else: 
            centroid_x = 0                 
            centroid_y = 0                 
            camera_range = 0                 
            motor_r = 0                 
            motor_l = 0  

        # Displaying co-ordinates & camera range 
        central = 'Centre:  X:%d : Y:%d ' % (centroid_x, centroid_y)             
        distance = 'Range:   %d cm' % (camera_range) 
        #cv2.putText(OriginalFrame, central,(20, 25), font, 0.65, (255,255,255), 1, cv2.LINE_AA) 
        #cv2.putText(OriginalFrame, distance,(20, 45), font, 0.65, (255,255,255), 1 , cv2.LINE_AA) 

        # Displaying Frames 
        cv2.imshow('Main Frame', OriginalFrame) 
        if cv2.waitKey(1) == 27: 
            break 

    video.release()     
    cv2.destroyAllWindows() 
if __name__ == "__main__": 
    main()