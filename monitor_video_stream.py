# import the necessary packages
import datetime, sys, os, argparse
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

def gap_seconds(t0, t1):
    string1 = f'{t0[0]}/{t0[1]}/{t0[2]}T{t0[3]}:{t0[4]}:{t0[5]}'
    string2 = f'{t1[0]}/{t1[1]}/{t1[2]}T{t1[3]}:{t1[4]}:{t1[5]}'
    a = datetime.datetime.strptime(string1, "%Y/%m/%dT%H:%M:%S")
    b = datetime.datetime.strptime(string2, "%Y/%m/%dT%H:%M:%S")
    gap = abs((a - b)).seconds
    return gap

now = datetime.datetime.now()
d1 = now.year, now.month, now.day, now.hour, now.minute, now.second
now = datetime.datetime.now()
d2 = now.year, now.month, now.day, now.hour, now.minute, now.second

g = gap_seconds(d1, d2)
print(g)

def loadAugImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    augDics = {}
    for imgPath in myList:
        filename = os.path.splitext(imgPath)[0]
        key = filename if filename == 'generic' else int(filename)
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
    return augDics

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = cv2.aruco.Dictionary_get(key)
    arucoParam = cv2.aruco.DetectorParameters_create()
    bboxs, ids, rejected = cv2.aruco.detectMarkers(imgGray, arucoDict, parameters = arucoParam)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, bboxs)
    return [bboxs, ids]

def augmentAruco(bbox, ids, img, imgAug, drawId=True):
    tl = int(bbox[0][0][0]), int(bbox[0][0][1])
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0,0], [w,0], [w,h], [0,h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int), (0,0,0))
    imgOut = img + imgOut
    if drawId:
        cv2.putText(imgOut, str(ids), tl, cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    return imgOut

def monitor_video_stream(rtsp_url, arucoDict, arucoParams, GAP_SECONDS=20):
    seendict = {}
    now = datetime.datetime.now()
    timestamp = f'{now.year}.{now.month}.{now.day}T{now.hour}:{now.minute}'
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
    arucoParams = cv2.aruco.DetectorParameters_create()
    vs = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # Set output frame height and width
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width  = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Set playback FPS to 24
    fps = 24
    fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
    writer = cv2.VideoWriter(f"records/record-{timestamp}.avi", fourcc, fps, (width, height))
    seenlog = open(f"records/record-{timestamp}.log","w")
    augDics = loadAugImages("images")

    while True:
        flag, frame = vs.read()
        # writer.write(frame)
        # frame = imutils.resize(frame, width=1000)
        arucoFound = findArucoMarkers(frame, markerSize=5, totalMarkers=250)
        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDics.keys():
                    frame = augmentAruco(bbox, id, frame, augDics[int(id)])
                else:
                    frame = augmentAruco(bbox, id, frame, augDics['generic'])
                ts = datetime.datetime.now()
                tsarr = ts.year,ts.month,ts.day,ts.hour,ts.minute,ts.second
                ARcode = id[0]
                if ARcode in seendict:
                    lastseen = seendict[ARcode][-1]
                    if gap_seconds(lastseen, tsarr) > GAP_SECONDS:
                        seendict[ARcode].append(tsarr)
                        seenlog.write(f'{ARcode},{tsarr[0]}/{tsarr[1]}/{tsarr[2]}T{tsarr[3]}:{tsarr[4]}:{tsarr[5]}\n')
                else:
                    seendict[ARcode] = [tsarr]
                    seenlog.write(f'{ARcode},{tsarr[0]}/{tsarr[1]}/{tsarr[2]}T{tsarr[3]}:{tsarr[4]}:{tsarr[5]}\n')
        # show the output frame
        cv2.imshow("Frame", frame)
        writer.write(frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # cleanup
    vs.release()
    writer.release()
    seenlog.close()
    cv2.destroyAllWindows()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_6X6_1000", help="tag type")
args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
    sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
monitor_video_stream('rtsp://192.168.2.134:5554/camera', arucoDict, arucoParams)
