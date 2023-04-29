# Image-recognition
This is a capstone project.
How it work:
  A basic advertisement loop that keeps displaying the advertisements. Once a brand logo from 'brand_list' is detected, it breaks out from the basic advertisement loop and display the detected advertisement. When the detected brand advertisement is done, program goes back to basic advertisement loop where it left off. 
  This is using CLIP - ViT-B-32-quickgelu and laion400m_e32
How to use:
  There is a same comment at the end of the loop.

  you have to run this in python 3.10 (3.11 is the latest). 
  To run, download libraries first, on CMD, 'pip install open_clip_torch' 'pip install torch' 'pip install opencv-python' 'pip install numpy' 
  On VS code, open terminal change Power Shell to CMD, change the directory and 'python main.py' to run the code
  when first time running the code on a device, it will download some pre-trained variables (only first time on a device)
  As it is running it will save the brand name on 'brand_queue'.
  press 'esc' button to exit the program. then it will show you what is in 'brand_list' on terminal
