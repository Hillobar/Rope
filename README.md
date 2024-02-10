![Screenshot 2024-02-10 091752](https://github.com/Hillobar/Rope/assets/63615199/dd8ab00b-d55f-4196-a50b-f2a326fba83a)

Rope implements the insightface inswapper_128 model with a helpful GUI.
### [Discord](https://discord.gg/EcdVAFJzqp)

### [Donate](https://www.paypal.com/donate/?hosted_button_id=Y5SB9LSXFGRF2)

### [Wiki with install instructions and usage](https://github.com/Hillobar/Rope/wiki)

### [Demo Video (Rope-Ruby)](https://www.youtube.com/watch?v=4Y4U0TZ8cWY)

### ${{\color{Goldenrod}{\textsf{Last Updated 2024-02-10 11:42 AM PST}}}}$ ###
### ${{\color{Goldenrod}{\textsf{Welcome to Rope-Opal!}}}}$ ###
### ${{\color{Red}{\textsf{Please grab the latest yoloface model from the link in the wiki!}}}}$ ###

![Screenshot 2024-02-10 104718](https://github.com/Hillobar/Rope/assets/63615199/4b2ee574-c91e-4db2-ad66-5b775a049a6b)

### Features: ###
* Lightning speed face swapping with all the features
* Upscalers
* Likeness modifiers
* Orientation management
* Masks: borders, differentials, auto occlusion, face parsers, text-based masking - all with strength adjustments and blending settings
* Mask view to evaluate masks directly
* Source face merging and saving
* Swap images or videos
* Auto save filename generation
* Dock/Undock the video player
* Real-time player
* Segment recording
* Fine tune your video ahead of time by creating image setting markers at specific frames.
* Lightening fast!

### Updates for Rope-Opal: ###
* This next version focuses on the UI. It's completely overhauled and finally looks more like a modern app. Lots of useability improvements and room to add new features.
* Can now set strength to zero. This effectively turns off the swapper in the render pipeline so you can apply the rest of the options to the original image (e.g., upscale the original face w/o swapping)
* Recording library can be set to FFMPEG or OPENCV
* Real-time audio is now available while previewing. Performace that renders slower than the frame rate will cause audio lag
* The Differencing fuction has been reworked into the pipeline to produce better results. Consequently it currently does not show up in the mask preview.
* Wrestled back some VRAM from the Ruby upgrades
* Faster loading of some models. Upcoming releases will do furhter optimizations
* Adjusted the use of filtering and antialiasing
* Yolov8 added as a face detection model selection. FF is having good results with it, so looking forward to hearing your thoughts on its behavior in Rope
* Scrollbars!
* Save/load paramters, and reset to defaults. Rope will auto-load your saved paramters when launched.
* Restorers (GFPGAN, etc) now have option to choose the detection alignment method. You can trade speed vs fidelity vs texture. This includes the original Rope method that, although flawed, maintain the face textures.
* Detection score. Adjust how aggressive Rope is at finding faces.
* Added detailed help text in the lower left when hovering over UI elements.
* Added reverse, forward and beginning to timeline control.

### Some Feature Still need to be re-implmented from Rope-Ruby. They'll be added back in the next updates. ###
* Image swapping
* Stop markers
* Framerate stats while playing
* Global hotkeys for moving the timeline
* Clear memory
* Ongoing interface maturation

### Performance:  ###
Machine: 3090Ti (24GB), i5-13600K

<img src="https://github.com/Hillobar/Rope/assets/63615199/3e3505db-bc76-48df-b8ac-1e7e86c8d751" width="200">

File: benchmark/target-1080p.mp4, 2048x1080, 269 frames, 25 fps, 10s
Rendering time in seconds:
| Option | Crystal | Sapphire | Ruby | Opal |
| --- | --- | --- | --- | --- |
| Only Swap | 7.3 | 7.5 | 4.4 | 4.3 |
| Swap+GFPGAN | 10.7 | 11.0 | 9.0 | 9.8 |
| Swap+Codeformer | 12.4 | 13.5 | 11.1 | 11.1 |
| Swap+one word CLIP | 10.4 | 11.2 | 9.1 | 9.3 |
| Swap+Occluder | 7.8 | 7.8 | 4.4 | 4.7 |
| Swap+MouthParser | 13.9 | 12.1 | 5.0 | 4.9 |

### Disclaimer: ###
Rope is a personal project that I'm making available to the community as a thank you for all of the contributors ahead of me.
I've copied the disclaimer from [Swap-Mukham](https://github.com/harisreedhar/Swap-Mukham) here since it is well-written and applies 100% to this repo.
 
I would like to emphasize that our swapping software is intended for responsible and ethical use only. I must stress that users are solely responsible for their actions when using our software.

Intended Usage: This software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. I encourage users to explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.

Ethical Guidelines: Users are expected to adhere to a set of ethical guidelines when using our software. These guidelines include, but are not limited to:

Not creating or sharing content that could harm, defame, or harass individuals. Obtaining proper consent and permissions from individuals featured in the content before using their likeness. Avoiding the use of this technology for deceptive purposes, including misinformation or malicious intent. Respecting and abiding by applicable laws, regulations, and copyright restrictions.

Privacy and Consent: Users are responsible for ensuring that they have the necessary permissions and consents from individuals whose likeness they intend to use in their creations. We strongly discourage the creation of content without explicit consent, particularly if it involves non-consensual or private content. It is essential to respect the privacy and dignity of all individuals involved.

Legal Considerations: Users must understand and comply with all relevant local, regional, and international laws pertaining to this technology. This includes laws related to privacy, defamation, intellectual property rights, and other relevant legislation. Users should consult legal professionals if they have any doubts regarding the legal implications of their creations.

Liability and Responsibility: We, as the creators and providers of the deep fake software, cannot be held responsible for the actions or consequences resulting from the usage of our software. Users assume full liability and responsibility for any misuse, unintended effects, or abusive behavior associated with the content they create.

By using this software, users acknowledge that they have read, understood, and agreed to abide by the above guidelines and disclaimers. We strongly encourage users to approach this technology with caution, integrity, and respect for the well-being and rights of others.

Remember, technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of deep fake technology for the betterment of society.



  
