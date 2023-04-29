import numpy as np
import cv2
import torch
from tkinter import *
from PIL import Image
from PIL import ImageTk
from win32api import GetSystemMetrics
import open_clip
import keyboard

# obtain system screen sizes for displaying images

width = GetSystemMetrics(0)
height = GetSystemMetrics(1)

# changing width and height to strings for later use

sWidth = str(width)
sHeight = str(height)

# import pictures and make them variables

Welcome = Image.open(r"C:\Users\kunch\OneDrive\Desktop\Capstone\Ads\Welcome.jpg")
Welcome = Welcome.resize((width,height),Image.ANTIALIAS)

Disney = Image.open(r"C:\Users\kunch\OneDrive\Desktop\Capstone\Ads\Disney.jpg")
Disney = Disney.resize((width,height),Image.ANTIALIAS)

Mcdonalds = Image.open(r"C:\Users\kunch\OneDrive\Desktop\Capstone\Ads\mcdonalds.png")
Mcdonalds = Mcdonalds.resize((width,height),Image.ANTIALIAS)

King = Image.open(r"C:\Users\kunch\OneDrive\Desktop\Capstone\Ads\Burger_King.jpg")
King = King.resize((width,height),Image.ANTIALIAS)

Tommy = Image.open(r"C:\Users\kunch\OneDrive\Desktop\Capstone\Ads\Tommy.jpg")
Tommy = Tommy.resize((width,height),Image.ANTIALIAS)

Adidas = Image.open(r"C:\Users\kunch\OneDrive\Desktop\Capstone\Ads\Adidas_Ad.jpg")
Adidas = Adidas.resize((width,height),Image.ANTIALIAS)

Nike = Image.open(r"C:\Users\kunch\OneDrive\Desktop\Capstone\Ads\Nike_Ad.jpg")
Nike = Nike.resize((width,height),Image.ANTIALIAS)

Under_Armour = Image.open(r"C:\Users\kunch\OneDrive\Desktop\Capstone\Ads\Under_Armour_Ad.png")
Under_Armour = Under_Armour.resize((width,height),Image.ANTIALIAS)

Toledo = Image.open(r"C:\Users\kunch\OneDrive\Desktop\Capstone\Ads\utoledo.jpg")
Toledo = Toledo.resize((width,height),Image.ANTIALIAS)

#load OpenAI CLIP model and tokenizer architecture, and load pretrained model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

brand_list = ["nike logo", "adidas logo", "under armour logo", "university of toledo", "not a logo"]

ads = [Nike, Adidas, Under_Armour, Toledo, Welcome]	# Array of known ads
brand_queue = []
current_display = [Welcome, Disney, Mcdonalds, King, Tommy]

displayOn = False
win = Tk()
win.attributes('-fullscreen', True)
canvas = Canvas(win, width= width, height= height)
count = 0

# Function for displaying the images
def displayImage(x,y):
    global displayOn
    global win
    global canvas
    if (displayOn):
        canvas.delete("all")

    win.geometry(sWidth+"x"+sHeight)

    canvas.pack()

    img = ImageTk.PhotoImage(x[y])

    canvas.create_image(0,0,anchor=NW,image=img)
    win.after(2000,lambda:win.quit())
    win.mainloop()
    
    displayOn = True

def from_ndarray_to_PIL(ndarray_img):
  #Converts ndArray image into PIL
  #image webcam ndArray
  return Image.fromarray(ndarray_img)

def classify_brand(pil_img):
  #Takes PIL image and gets it's brand name. 
  image = preprocess(pil_img).unsqueeze(0)
  text = tokenizer(brand_list)
  with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

  return brand_list[np.argmax(text_probs.cpu().detach().numpy())]

cap = cv2.VideoCapture(1)


while 1:
    ret, img = cap.read() # numpy array 
    brand_name = classify_brand(from_ndarray_to_PIL(img))
    
    #append to brand_name array so they do not append same brand in a row
    if brand_name != "not a logo" :
      if len(brand_queue) != 0:
        if brand_name != brand_queue[-1]:
          brand_queue.append(brand_name)
      else:
        brand_queue.append(brand_name)
    
    i = 0
    while i < len(brand_list): # checks if Scan is in the known brands index
        if brand_name == brand_list[i]:
            if brand_name != "not a logo":
              displayImage(ads,i)
        i += 1
    
    # display image subprogram
    displayImage(current_display,count) 
    # keeps track of image display
    if (count)+1 == len(current_display): 
        count = 0
    else:
        count += 1

    #print(f"Output: {brand_name}")
    
    """
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    """
    if keyboard.is_pressed('esc'):
      break

cap.release()
cv2.destroyAllWindows()

#enumerate keep track of the number of iterations in a loop.
#when program quit, it prints the brand_queue
for i, name in enumerate(brand_queue):
  print(f"{i+1} :: {name}")

  """
recognition part is done and shared in Onedrive. 
you have to run this in python 3.10 (3.11 is the latest). 
To run, download libraries first, on CMD, 'pip install open_clip_torch' 'pip install torch' 'pip install opencv-python' 'pip install numpy' 
On VS code, open terminal change Power Shell to CMD, change the directory and 'python main.py' to run the code
when first time running the code on a device, it will download some pre-trained variables (only first time on a device)
As it is running it will save the brand name on 'brand_queue'.
press 'esc' button to exit the program. then it will show you what is in 'brand_queue' on terminal
  """