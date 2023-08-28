# Rope![Screenshot 2023-08-28 100045](https://github.com/Hillobar/Rope/assets/63615199/06dc477d-d479-4d70-9ff3-03b7d8f12b7b)

Rope implements the insightface inswapper_128 model with a helpful GUI.
### New Discord link: ###
[Discord](https://discord.gg/EcdVAFJzqp)

### Features: ###
* Incredible features and fast workflow
* High performance
* Real-time video player
* Occlusion functions

### Upcoming changes for Rope - Crystal: ###
* Mouse Scroll on Target/Source Faces and Target Videos
* Improvements in workflow efficiency
* Fixed some bugs with Recording
* ?
* ?
* ? 

### Known bugs: ### 
* Stop video playback before loading a new video, or bork

### Preview: ###
![Screenshot 2023-08-05 154156](https://github.com/Hillobar/Rope/assets/63615199/921698ab-af0e-43ca-b669-a2b2537d5c0f)
### Getting Started: ###
![Screenshot 2023-08-05 152851](https://github.com/Hillobar/Rope/assets/63615199/68b4ec4e-615f-4fd6-9215-f5a2ae8314b4)
### Features: ###
![Screenshot 2023-08-05 152835](https://github.com/Hillobar/Rope/assets/63615199/4e64237e-7d0f-4a83-a738-64b0df206766)

### Disclaimer: ###
Rope is a personal project that I'm making available to the community as a thank you for all of the contributors ahead of me. I don't have time to troubleshoot or add requested features, so it is provided as-is. Don't look at this code for example of good coding practices. I am primarily focused on performance and my specific use cases. There are plenty of ways to bork the workflow. Please see how to use below.

### Install: ###
Note: It's only configured for CUDA (Nvidia)
* Set up a local venv
  * python.exe -m venv venv
* Activate your new venv
  * .\venv\Scripts\activate
* Install requirements
  * .\venv\Scripts\pip.exe install -r .\requirements.txt
* Place [GFPGANv1.4.onnx](https://github.com/Hillobar/Rope/releases/download/Space_Worm/GFPGANv1.4.onnx), [inswapper_128_fp16.onnx](https://github.com/Hillobar/Rope/releases/download/Space_Worm/inswapper_128.fp16.onnx), and [occluder.ckpt](https://github.com/Hillobar/Rope/releases/download/Space_Worm/occluder.ckpt) in the root directory
* Do this if you've never installed roop or Rope (or any other onnx runtimes):
  * Install CUDA Toolkit 11.8
  * Install dependencies:
  * pip uninstall onnxruntime onnxruntime-gpu
  * pip install onnxruntime-gpu==1.15.1
* Double-click on Rope.bat!

### To use: ###
* Run Rope.bat
* Set your Target Video, Source Faces, and Video Output folders
  * Buttons will be gold if they are not set
  * Only places videos or images in the respective folders. Other files my bork it
  * Rope creates a JSON file to remember your last set paths
  * I like to keep my folders <20 or so items. Helps to organize and reduces load times
* Click on the Load Models button to initialize Rope
* Select a video to load it into the player
* Find Target Faces
  * Adds all faces in the current frame to the Found Faces pane
  * If a Face is already Found and in the pane, it won't re-add it
* Click a Source Face
  * Source Face number will appear
* Select a Target Face
  * Target Faces will show assignment number to the Source Face number
  * Toggle a Target Face to unselect and reassign to currently selected Source Face
* Continue to select other Source Faces and assign them to Target Faces
* Click SWAP to enable face swapping
* Click PLAY to play
* Click REC to arm recording
  * Click PLAY to start recording using the current settings
  * Click PLAY again to stop recording, otherwise it will record to the end of the Target Video
* Toggle GFPGAN, adjust blending amount
* Toggle Diffing, adjust blending amount
* Lower the threshhold if you have multiple Source Faces assigned and they are jumping around. You can also try Clearing and Finding new Target Faces (disable SWAP first)
* Modify the Masking boundaries
* Use CLIP to identify objects to swap or not swap (e.g Pos: face, head; Neg: hair, hand), adjust the gain of the words, and set the blur amount around the items
* Change # threads to match your GPU memory (24GB ~9 threads with GFPGAN on, more threads w/o GFPGAN)
  * Start with the lowest you think will run and watch your GPU memory.
  * Once you allocate memory by increasing # threads, you can't un-allocate it by reducing # threads. You will need to restart Rope.
* In general, always stop the video before changing anything. Otherwise, it might bork. Reassigning faces is okay
* If it does bork, reload the video (reclick on it). If that doesn't work you'll need to restart

### Changelog ###
### Changes for Rope - Space Worm: ###
* Updated video rendering to use Target Video parameters
* Mousewheel scroll on the time bar to control frame position
* Added an occluder model (experimental, very fast, make sure you download the new model-link below)
* Greatly increased performance for larger videos/multiple faces
* CLIP crashing fixed. Add as many words as you like!
* Detachable video preview
* Fixed most bugs related to changing options while playing. Adjust setting on the fly!
* GFPGAN now renders up to 512x512
* Status bar (still adding features to this)
  
