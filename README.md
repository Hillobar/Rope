# Rope
Rope ikmplements the insightface inswapper_128 model with a helpful GUI.

Disclaimer:

Also:
Rope is a personal project that I'm making available to the community as a thank you for all of the ocntributors ahead of me. I don't have time to troubleshoot or add requested features, so it is provided as-is. Don't look at this code for example of good coding practices. I am primarily focused on performance and my specific use cases. There are plenty of ways to bork the workflow. Please see how to use below.

Features:
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

GUI:
![Rope](https://github.com/Hillobar/Rope/assets/63615199/d4120e5d-6fd4-4c01-88ac-e236d0379832)

To use:
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
*  
