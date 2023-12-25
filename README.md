![Screenshot 2023-12-22 211725](https://github.com/Hillobar/Rope/assets/63615199/97a57957-fb30-4329-b8f6-adbfa96203ab)

Rope implements the insightface inswapper_128 model with a helpful GUI.
### [Discord](https://discord.gg/EcdVAFJzqp)

### [Donate](https://www.paypal.com/donate/?hosted_button_id=Y5SB9LSXFGRF2)

### [Wiki](https://github.com/Hillobar/Rope/wiki)

### [Demo Video](https://www.youtube.com/watch?v=4Y4U0TZ8cWY)

### ${{\color{Goldenrod}{\textsf{Last Updated 2023-12-22 21:42 PST}}}}$ ###
note: you need to download the latest GFPGAN model for Ruby. Link is in the wiki

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

### Updates for Rope-Ruby: ###
* Almost double the performance of previous Rope! Twice as fast! Half the time! Most of the effort for Ruby focuses on huge performance gains. Enjoy the speed!
* Much faster GFPGAN
* Occluder mask size can now be adjusted
* Experimental features added to make adjustments to face swap region placement and face scale.

### Known Bugs: ###
- When using Markers, the frames before the first marker will use parameters from the last settings in your options. Not sure if it is a true bug, but best way to deal with this is to create a marker at the first frame.
- Pressing 'a' at frame 0 will stop playback after a few seconds of play
- Lower resolution videos (less than 512 pixels in one of the directions) will result in only part of the face becoming swapped.

### Performance:  ###
Machine: 3090Ti (24GB), i5-13600K

<img src="https://github.com/Hillobar/Rope/assets/63615199/3e3505db-bc76-48df-b8ac-1e7e86c8d751" width="200">

File: benchmark/target-1080p.mp4, 2048x1080, 269 frames, 25 fps, 10s
Rendering time in seconds:
| Option | Crystal | Sapphire | Ruby |
| --- | --- | --- | --- |
| Only Swap | 7.3 | 7.5 | 4.4 |
| Swap+GFPGAN | 10.7 | 11.0 | 9.0 |
| Swap+Codeformer | 12.4 | 13.5 | 11.1 |
| Swap+one word CLIP | 10.4 | 11.2 | 9.1 |
| Swap+Occluder | 7.8 | 7.8 | 4.4 |
| Swap+MouthParser | 13.9 | 12.1 | 5.0 |

### Preview (from Rope-Crystal): ###
![Screenshot 2023-12-22 212639](https://github.com/Hillobar/Rope/assets/63615199/384fd63a-b870-4714-a137-d27e31560433)


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



  
