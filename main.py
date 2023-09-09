print("Setting Up")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
from utlis import *
from show_all_imag import stackImages
import sudukoSolver

##################

path = "img/3.jpg"
hImg = 450
wImg = 450
model = intializePredectinoModel()
###################
img = cv2.imread(path)
img = cv2.resize(img, (wImg,hImg))
imgBlack = np.zeros((wImg,hImg,3),np.uint8)
imgThreshold = preProcess(img)
########################
imgContours = img.copy()
imgBigContour = img.copy()
contours , hierarchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours,contours,-1,(0,255,0),3)
########################
biggest, maxArea = biggestContours(contours)
if  biggest.size != 0:
    biggest = reorder(biggest)
    cv2.drawContours(imgBigContour,biggest,-1,(0,255,0),10)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[wImg,0],[0,hImg],[wImg,hImg]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarpColored = cv2.warpPerspective(img , matrix,(wImg,hImg))
    imgDetectedDigits = imgBlack.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored , cv2.COLOR_BGR2GRAY)

    #########
    imgSolvedDigits = imgBlack.copy()
    boxes = splitBoxes(imgWarpColored)
    #print(boxes[0].shape)
    # cv2.imshow("box", boxes[7])
    numbers = getPredection(boxes,model)
    print(numbers)
    imgDetectedDigits = displayNumbers(imgDetectedDigits,numbers,color=(255,0,255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)
    print(posArray)
    #### 5. FIND SOLUTION OF THE BOARD
    board = np.array_split(numbers, 9)
    print(board)
    try:
        sudukoSolver.solve(board)
    except:
        pass
    print(board)
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList * posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

    # #### 6. OVERLAY SOLUTION
    pts2 = np.float32(biggest)  # PREPARE POINTS FOR WARP
    pts1 = np.float32([[0, 0], [wImg, 0], [0, hImg], [wImg, hImg]])  # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (wImg, hImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
    imgDetectedDigits = drawGrid(imgDetectedDigits)
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
    stackedImage = stackImages(imageArray, 1)
    stackedImage = cv2.resize(stackedImage, (1000, 500))
    cv2.imshow('Stacked Images', stackedImage)

else:
    print("No Sudoku Found")

cv2.waitKey(0)










