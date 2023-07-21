# Rope
Rope implements the insightface inswapper_128 model with a helpful GUI.

### Discord link: ###
[Discord](https://discord.gg/HAKNAxZT)

### Upcoming changes (Will release 7/22): ###
* Modified inswapper.onnx file to be faster (13 ms -> 9 ms per swap on my 3090)
* Added txt2mask to specify masked and unmasked areas on and around the face
* Record button - press record and anything that you play is recorded to a video file (good for just capturing segments)
* Updated GUI
* Updated the video creation logic to fix a bug
* Bug: fixed bad colors in skin and GFPGAN 
  
### Disclaimer: ###
Rope is a personal project that I'm making available to the community as a thank you for all of the ocntributors ahead of me. I don't have time to troubleshoot or add requested features, so it is provided as-is. Don't look at this code for example of good coding practices. I am primarily focused on performance and my specific use cases. There are plenty of ways to bork the workflow. Please see how to use below.

### Features: ###
* Real-time video player
* Optimized model paths (runs >30fps with GFPGAN on 3090Ti)
* Resizeable window
* Load, view, and select Source Videos and Source Faces from specified folders
* Identify Target Faces from current frame
* Map multiple Source Faces to mutiple Target Faces
* GFPGAN with blending
* Diffing with blending
* Adjust Face boudaries to match Source and Target Faces, with blending
* Set threads
* Set face matching threshhold
* Create videos with current settings
* Created videos add audio and compress file size
  
### Install: ###
Note: It's only configured for CUDA (Nvidia)
* Set up a local venv
* Install requirements
* Place [GFPGANv1.4.onnx](https://github.com/Hillobar/Rope/releases/download/Model_files/GFPGANv1.4.onnx)  and [inswapper_128.onnx](https://github.com/Hillobar/Rope/releases/download/Model_files/inswapper_128.onnx) in the root directory
* If you're already set up for roop, you probably don't need to do the next steps:
  * Install CUDA Toolkit 11.8
  * Install dependencies:
  * pip uninstall onnxruntime onnxruntime-gpu
  * pip install onnxruntime-gpu==1.15.1

### GUI: ###
![Rope](https://github.com/Hillobar/Rope/assets/63615199/bbb60010-0e36-40f3-9069-638d45a07515)


### To use: ###
* Run Rope_v3.bat
* Create your Video and Faces folders, add stuff, and Set them
  * Only places videos or images in the respective folders. Other files my bork it
  * Rope creates a JSON file to remember your last set paths
  * I like to keep my folders <20 or so videos. Helps to organize and reduces load times
* Load Videos
  * Might take a sec or two depending on # of videos
  * Videos are displayed on the scrollable pane
  * Click video to load into player
  * Play, scrub
* Load Source Faces
  * If you haven't clicked Find Faces yet, Rope will load up a model, which will take a few seconds. Subsequent loads (loading a new folder in the same session) will be almost instant
* Find Faces
  * Adds all faces in the current frame to the Found Faces pane
  * If a Face is already Found and in the pane, it won't re-add it
* Click a Source Face
  * Source Face number will appear
* Select a targeted Found Face
  * Found Face will show assignment number to Source Face number
  * Toggle the targeted Found Face to unselect
* Continue to select other Source Faces and assign them to Found Faces
* Toggle Swap (loads the model the first time)
* Toggle GFPGAN
* Modify the boundaries and boundary blending (affects all faces)
* Change # threads to match your GPU memory (24GB ~9 threads with GFPGAN on, more threads w/o GFPGAN)
  * Start with the lowest you think will run and watch your GPU memory.
  * Once you allocate memory by increasing # threads, you can't un-allocate it by reducing # threads. You will need to restart Rope.
* Create Videos with the curent settings. Wait for the audio to be added and the final video to be re-encoded
* Reduce Face matching if you see some of your assignments switching to other faces. (finicky)
* In general, always stop the video before changing anything. Otherwise, it will likely bork.
* If it does bork, reload the video (reclick on it). If that doesn't work you'll need to restart
  
