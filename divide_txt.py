import cv2
import os
import sys
import shutil

LineSet = [] # Line List



#large = cv2.imread("originImage\\font1.png") # 130, (6,6), 2 , 0.3
large = cv2.imread("originImage\\font3.png") # 150  0.3

#large = cv2.imread("originImage\\hand1.jpg") # 100 0.3


lineCount = 0
R_Threshold = 0.3
point_Threshold = 0.5

image = large
copy = image.copy()

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale

#_,thresh = cv2.threshold(gray,20,255,cv2.THRESH_BINARY_INV) # threshold   cv2.THRESH_BINARY cv2.THRESH_BINARY_INV 150
_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold   cv2.THRESH_BINARY cv2.THRESH_BINARY_INV 150


# delete all files in Lines folder
if os.path.exists('getBlock_Image'):
    shutil.rmtree('getBlock_Image')

# make Lines folder
os.makedirs('getBlock_Image')

cv2.imwrite('getBlock_Image\\gray.jpg', gray)
cv2.imwrite('getBlock_Image\\thre.jpg', thresh)

# delete all files in Lines folder
if os.path.exists('Blocks'):
    shutil.rmtree('Blocks')

# make Lines folder
os.makedirs('Blocks')


Height, Width = thresh.shape



def getLine() :
    print("Start Get Line")
    print("W : ", str(Width)  + " H : " + str(Height))

    global lineCount
    lookStartY = 0
    lookEndY = 1

    state = lookStartY
    posY = 1

    startY = 1
    endY = 1

    while( posY < Height) :
        if(state == lookStartY) :
            #print("look Start Y")

            flag = False
            for x in range(1, Width-1):
                pixel = thresh[posY, x]

                if(pixel == 255):
                    flag = True
                    startY = posY

                    lineCount+=1

                    #print("#" + str(lineCount) + "  Start Y : " + str(startY))
                   # cv2.line(image, (1, startY), (Width-1, startY), (0, 0, 255), 2)
                    break

            if(flag is True):
                state = lookEndY

        if(state == lookEndY):
           # print("look End Y")

            flag = True
            for x in range(1, Width - 1):
                pixel = thresh[posY, x]
                #글자 조금이라도 걸리면
                if(pixel == 255):
                    flag = False

            if(flag is True):
                endY = posY
               # print("#" + str(lineCount) + "  End Y : " + str(endY))
               # cv2.line(image, (1, endY), (Width - 1, endY), (0, 255, 0), 2)

                line = [startY, endY]
                LineSet.append(line)

                state = lookStartY
        posY += 1


    print(" Line Count : " + str(lineCount))


def getCharPerLine(lineNum):
    #print(" get PEr Line Start Look " + str(lineNum + 1))

    START_Y = LineSet[lineNum][0]
    END_Y = LineSet[lineNum][1]

    H = END_Y - START_Y

    lookStartX = 0
    lookEndX = 1
    state = lookStartX

    posX = 1

    Left = 1
    Right = 1

    space_Count = 0
    char_NUM = 1
    IDX = 1;

    # LX., RX , TY , DY, Space
    preCharPoint = [0, 0, 0, 0, 0]
    startFlag = -1

    while(posX < Width):
        if(state == lookStartX):
            flag = False

            for y in range (START_Y, END_Y-1):
                pixel = thresh[y, posX]

                if(pixel == 255):
                    flag = True
                    break

            if(flag is True):
                state = lookEndX
                Left = posX

                # 앞 글자와 얼마나 뛰어져 있는지 판단해야함
                space_Count = int((Left - Right) / 10)
                #print("distance : " + str(space_Count))

            else :
                posX+=1


        if(state == lookEndX):
            flag = True
            for y in range(START_Y, END_Y-1):
                pixel = thresh[y,posX]

                if(pixel == 255):
                    flag = False

            if(flag is False):
                posX+=1

            #그대로이면 끝난거
            else:
                state = lookStartX

                Right = posX
                realStartY = START_Y
                realEndY = END_Y

                charH = END_Y - START_Y
                charW = Right - Left
                charR = charW / charH

              #  print(" c R : " + str(charR))

                # Y 시작값 탐색
                for y in range(START_Y, END_Y - 1):
                    flag = False
                    for x in range(Left, Left):
                        pixel = thresh[y, x]
                        if (pixel == 255):
                            flag = True
                            break

                    if (flag is True):
                        realStartY = y - 4
                        break;


                # Y 끝값 탐색
                for y in range(END_Y - 1, realStartY + 1, -1):
                    flag = False
                    for x in range(Left, Right):
                        pixel = thresh[y, x]
                        if (pixel == 255):
                            flag = True

                    if (flag is True):
                        realEndY = y + 4
                        break

                charH2 = realEndY - realStartY
                charW2 = Right - Left
                charR2 = charW2 / charH2

                print(" c R 2: " + str(charR2) + " c R : " + str(charR))

                # 조금의 여분을 남김

                # 너무 작고 앞글자와 거리가 좁으면 이전 글자와 합친다.
                if (charR2 < R_Threshold):
                    preCharPoint[1] = Right

                    if (preCharPoint[2] > realStartY):
                        preCharPoint[2] = realStartY

                    if (preCharPoint[3] < realEndY):
                        preCharPoint[3] = realEndY

                # 원본 이미지에 짜른 영역 표시
                #cv2.line(image, (Left, START_Y), (Left, END_Y), (0, 0, 255), 1)
                #cv2.line(image, (Right, START_Y), (Right, END_Y), (0, 0, 255), 1)
                #cv2.line(image, (Left, START_Y), (Right, START_Y), (0, 0, 255), 1)
                #cv2.line(image, (Left, END_Y), (Right, END_Y), (0, 0, 255), 1)

                # 이전 글자가 있으면
                if (startFlag != -1):
                    lineIdx = '0'
                    charIdx = '0'

                    # 01, 02 ,03  ..... 10, 11 순으로 저장
                    if ((lineNum + 1) < 10):
                        lineIdx = str(0) + str(lineNum + 1)

                    if ((lineNum + 1) > 9):
                        lineIdx = str(lineNum + 1)

                    if (char_NUM < 10):
                        charIdx = str(0) + str(char_NUM)

                    if (char_NUM > 9):
                        charIdx = str(char_NUM)

                    cutHeight = preCharPoint[3] - preCharPoint[2]
                    pointFlag = cutHeight / H
                    #print(" Ratio : " + str(pointFlag))

                    if (pointFlag < point_Threshold):
                        file_path = "Blocks\\" + lineIdx + "_" + charIdx + "_" + str(preCharPoint[4]) + "_0.png"

                    else:
                        file_path = "Blocks\\" + lineIdx + "_" + charIdx + "_" + str(preCharPoint[4]) + "_1.png"


                    roi = copy[preCharPoint[2]:preCharPoint[3], preCharPoint[0]:preCharPoint[1]]
                    cv2.imwrite(file_path, roi)

                    IDX += 1
                    char_NUM += 1

                    # 원본 이미지에 짜른 영역 표시
                    cv2.line(image, (preCharPoint[0], START_Y), (preCharPoint[0], END_Y), (0, 0, 255), 1)
                    cv2.line(image, (preCharPoint[1], START_Y), (preCharPoint[1], END_Y), (0, 0, 255), 1)
                    cv2.line(image, (preCharPoint[0], START_Y), (preCharPoint[1], START_Y), (0, 0, 255), 1)
                    cv2.line(image, (preCharPoint[0], END_Y), (preCharPoint[1], END_Y), (0, 0, 255), 1)

                    # 제대로 짤렸으면, 구한 값(현재 값)을 그대로 저장한다.
                    if (charR2 >= R_Threshold or (charR2 < R_Threshold and space_Count > 1)):
                        preCharPoint = [Left, Right, realStartY, realEndY, space_Count]

                    # 작게 짤려서 이전 꺼랑 합친 경우, 새로운 글자를 다시 찾도록 설정
                    if (charR2 < R_Threshold and space_Count < 2):
                        startFlag = -1

                # 처음이면 글자 영역 저장
                else:
                    startFlag = 0
                    preCharPoint = [Left, Right, realStartY, realEndY, space_Count]


    if (startFlag == 0):
        if ((lineNum + 1) < 10):
            lineIdx = str(0) + str(lineNum + 1)

        if ((lineNum + 1) > 9):
            lineIdx = str(lineNum + 1)

        if (char_NUM < 10):
            charIdx = str(0) + str(char_NUM)

        if (char_NUM > 9):
            charIdx = str(char_NUM)

        cutHeight = preCharPoint[3] - preCharPoint[2]
        pointFlag = cutHeight / H
        # print(" Ratio : " + str(pointFlag))

        if (pointFlag < point_Threshold):
            file_path = "Blocks\\" + lineIdx + "_" + charIdx + "_" + str(preCharPoint[4]) + "_0.png"

        else:
            file_path = "Blocks\\" + lineIdx + "_" + charIdx + "_" + str(preCharPoint[4]) + "_1.png"

        roi = copy[preCharPoint[2]:preCharPoint[3], preCharPoint[0]:preCharPoint[1]]
        cv2.imwrite(file_path, roi)

        cv2.line(image, (preCharPoint[0], START_Y), (preCharPoint[0], END_Y), (0, 0, 255), 1)
        cv2.line(image, (preCharPoint[1], START_Y), (preCharPoint[1], END_Y), (0,0, 255), 1)
        cv2.line(image, (preCharPoint[0], START_Y), (preCharPoint[1], START_Y), (0, 0, 255), 1)
        cv2.line(image, (preCharPoint[0], END_Y), (preCharPoint[1], END_Y), (0, 0, 255), 1)




def getCharacter():
    for idx in range(0, lineCount):
        getCharPerLine(idx)


getLine()
getCharacter()

'''
for i in range(0,lineCount):
    print("#" + str(i+1) + " Start : " + str(LineSet[i][0]) + " End : " + str(LineSet[i][1]))
'''
# write original image with added contours to disk
cv2.imshow('rects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()





