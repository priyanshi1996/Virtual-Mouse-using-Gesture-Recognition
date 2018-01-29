import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

mouse=Controller()
app=wx.App(False)

#screen size
(sx,sy)=wx.GetDisplaySize()
#changing image resolution
(camx,camy)=(320,240)

cam=cv2.VideoCapture(0)
cam.set(3,camx)#setting width
cam.set(4,camy)#setting height

lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])

kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

mouseLocOld=[0,0]
mouseLoc=[0,0]
openx,openy,openw,openh=(0,0,0,0)

DampingFactor=2# it should be greater than 1
#mouseLoc=mouseLocOld+(targetLoc-mouseLoc)//DampingFactor
pinchflag=0
while True:
    ret,img=cam.read()
    #img=cv2.resize(img,(320,220))
    #converting image from RGB to HSV,Where H represents hue means color
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    
    #morphology (Opening)
    #In opening kernel moves like a window to the whole image and all the white dots that
    #are on black background and are completely inside kernel are removed and background noise is removed
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    
    #morphology (Closing)
    #In closing kernel moves like a window to the whole image and all the black dots that
    #are on white background and are completely inside kernel are removed and background noise is removed
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    
    maskFinal=maskClose
    image,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    if(len(conts)==2):
        if(pinchflag==0):
            pinchflag=1
            mouse.release(Button.left)
        x1,y1,w1,h1=cv2.boundingRect(conts[0])
        x2,y2,w2,h2=cv2.boundingRect(conts[1])
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        cx1=x1+(w1//2)  #(centre of rectangle 1)
        cy1=y1+(h1//2)
        cx2=x2+(w2//2)  #(centre of rectangle 2)
        cy2=y2+h2//2
        cx=(cx1+cx2)//2; #centre of line
        cy=(cy1+cy2)//2;
        cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)
        cv2.circle(img,(cx,cy),2,(0,0,255),2)
        
        #mouseLoc=(sx-(cx*sx//camx), cy*sy//camy)
        mouseLoc[0]=mouseLocOld[0]+((cx)-mouseLocOld[0])//DampingFactor
        mouseLoc[1]=mouseLocOld[1]+((cy)-mouseLocOld[1])//DampingFactor
        mouse.position=(sx-(mouseLoc[0]*sx//camx), mouseLoc[1]*sy//camy)
        while mouse.position!=(sx-(mouseLoc[0]*sx//camx), mouseLoc[1]*sy//camy):
            pass
        mouseLocOld=mouseLoc
        #Here we are calculating the bounding rectangle of two objects and will see that
        #the difference between the area of box with two fingers and just one finger is too much
        #In that case we won't consider it a click.
        #If the difference is not too much then that means that two fingers slowly merges and 
        #we will consider it as a click
        openx,openy,openh,openw=cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))
        #cv2.rectangle(img,(openx,openy),(openx+openw,openy+openh),(255,0,0),2)
        
    elif(len(conts)==1):
        x,y,w,h=cv2.boundingRect(conts[0])
        if(pinchflag==1):
            pinchflag=0
            if(abs(w*h-openx*openy)/(w*h)<30):
                
                mouse.press(Button.left)
                openx,openy,openw,openh=(0,0,0,0)
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cx=x+w//2
        cy=y+h//2
        cv2.circle(img,(cx,cy),(w+h)//4,(0,0,255),2)
        
        mouseLoc[0]=mouseLocOld[0]+((cx)-mouseLocOld[0])//DampingFactor
        mouseLoc[1]=mouseLocOld[1]+((cy)-mouseLocOld[1])//DampingFactor
        mouse.position=(sx-(mouseLoc[0]*sx//camx), mouseLoc[1]*sy//camy)
        while mouse.position!=(sx-(mouseLoc[0]*sx//camx), mouseLoc[1]*sy//camy):
            pass
        
        
    
    #cv2.imshow("maskOpen",maskOpen)
    cv2.imshow("maskClose",maskClose)
    #cv2.imshow("mask",mask)
    cv2.imshow("img",img)
    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
    
cam.release()
cv2.destroyAllWindows()
    
