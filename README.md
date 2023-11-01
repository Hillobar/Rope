![Screenshot 2023-09-17 133025](https://github.com/Hillobar/Rope/assets/63615199/ebd12aa2-666c-49f5-a6f0-53f5eb1e5f52)

Rope implements the insightface inswapper_128 model with a helpful GUI.
### [Discord](https://discord.gg/EcdVAFJzqp)

### [Donate](https://www.paypal.com/donate/?hosted_button_id=Y5SB9LSXFGRF2)
Donations appreciated but not necessary 

### New [Wiki](https://github.com/Hillobar/Rope/wiki)
* New install instructions

### Features: ###
* Incredible features and fast workflow
* High performance
* Real-time video player
* Helpful functions

### (2023-09-17) Changelog for Rope - Crystal Shard: ###
**Note: Please check the wiki for installing new model files**

Fun Stuff:
* Added mousewheel function to Mouth Parser to adjust the size of the mask
* Added Codeformer as an enhancer option. Codeformer does a noticeably better job with skin textures, but runs slower. Right-click on the button to toggle Codeformer or GFPGAN. Note: Codeformer takes 15-30 secondfs to load the first time.

Boring Stuff:
* Mouth Parser and Occluder now use onnxruntime instead of PyTorch. Hopefully this will solve issues with AMD cpu users
* InsightFace libraries have been removed as a dependency
* Dependencies have been updated and aligned
* Performance increase
* Swapping is now automatically toggled off when dragging the timeline slider. this is to make it more responsive. It will automatically toggle the swap back on once you stop dragging if you had swap on to begin with.

Bug Fixes:
* Fixed bug when dragging the Video timeline. It can now be moved when playing
* Fixed several remaining bugs with recording 
* Fixed right click behavior on the video player slider

### Performance:  ###
Machine: 3090Ti (24GB), i5-13600K

<img src="https://github.com/Hillobar/Rope/assets/63615199/3e3505db-bc76-48df-b8ac-1e7e86c8d751" width="200">

File: benchmark/target-1080p.mp4, 2048x1080, 269 frames, 25 fps, 10s

| Option | Rendering Time [s] |
| --- | --- |
| Only Swap | 7.3 |
| Swap+GFPGAN | 10.7 |
| Swap+Codeformer | 12.4 |
| Swap+one word CLIP | 10.4 |
| Swap+Occluder | 7.8 |
| Swap+MouthParser | 13.9 |
| target-2160p.mp4 | 9.7 |

### Known bugs: ### 
* Stop video playback before loading a new video, or bork
* Nottifications on bottom sometime status early

### Preview: ###
![image](https://github.com/Hillobar/Rope/assets/63615199/fda0c05f-72a6-4935-a882-dc6d17cfc014)

### Disclaimer: ###
Rope is a personal project that I'm making available to the community as a thank you for all of the contributors ahead of me.
I've copied the disclaimer from [Swap-Mukham](https://github.com/harisreedhar/Swap-Mukham) here since it is well-written and applies 100% to this repo.
 
I would like to emphasize that our deep fake software is intended for responsible and ethical use only. I must stress that users are solely responsible for their actions when using our software.

Intended Usage: This software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. I encourage users to explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.

Ethical Guidelines: Users are expected to adhere to a set of ethical guidelines when using our software. These guidelines include, but are not limited to:

Not creating or sharing content that could harm, defame, or harass individuals. Obtaining proper consent and permissions from individuals featured in the content before using their likeness. Avoiding the use of this technology for deceptive purposes, including misinformation or malicious intent. Respecting and abiding by applicable laws, regulations, and copyright restrictions.

Privacy and Consent: Users are responsible for ensuring that they have the necessary permissions and consents from individuals whose likeness they intend to use in their creations. We strongly discourage the creation of content without explicit consent, particularly if it involves non-consensual or private content. It is essential to respect the privacy and dignity of all individuals involved.

Legal Considerations: Users must understand and comply with all relevant local, regional, and international laws pertaining to this technology. This includes laws related to privacy, defamation, intellectual property rights, and other relevant legislation. Users should consult legal professionals if they have any doubts regarding the legal implications of their creations.

Liability and Responsibility: We, as the creators and providers of the deep fake software, cannot be held responsible for the actions or consequences resulting from the usage of our software. Users assume full liability and responsibility for any misuse, unintended effects, or abusive behavior associated with the content they create.

By using this software, users acknowledge that they have read, understood, and agreed to abide by the above guidelines and disclaimers. We strongly encourage users to approach this technology with caution, integrity, and respect for the well-being and rights of others.

Remember, technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of deep fake technology for the betterment of society.



  
