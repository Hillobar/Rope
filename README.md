![Screenshot 2024-02-10 091752](https://github.com/Hillobar/Rope/assets/63615199/dd8ab00b-d55f-4196-a50b-f2a326fba83a)

Rope implements the insightface inswapper_128 model with a helpful GUI.


### [安装及使用说明](https://github.com/Hillobar/Rope/wiki)

### [Demo Video (Rope-Ruby)](https://www.youtube.com/watch?v=4Y4U0TZ8cWY)

### ${{\color{Goldenrod}{\textsf{Last Updated 2024-02-27}}}}$ ###
### ${{\color{Goldenrod}{\textsf{Welcome to Rope-Opal!}}}}$ ###
### ${{\color{Red}{\textsf{Please grab the latest yoloface model from the link in the wiki!}}}}$ ###

![Screenshot 2024-02-10 104718](https://github.com/Hillobar/Rope/assets/63615199/4b2ee574-c91e-4db2-ad66-5b775a049a6b)

### 功能特点： ###
* 闪电般速度的面部置换功能
* 全面图像增强器
* 相似度调整器
* 方向管理
* 遮罩功能：边缘、差异、自动遮挡、面部解析、基于文本的遮罩——所有这些都带有强度调整和混合设置
* 遮罩视图可以直接评估遮罩效果
* 合并并保存源面孔
* 交换图像或视频
* 自动生成保存文件名
* 视频播放器的停靠/分离
* 实时播放器
* 段落录制
* 通过在特定帧创建图像设置标记，提前微调视频。

### 性能:  ###
机器配置: 3090Ti (24GB), i5-13600K

<img src="https://github.com/Hillobar/Rope/assets/63615199/3e3505db-bc76-48df-b8ac-1e7e86c8d751" width="200">

文件： benchmark/target-1080p.mp4，2048x1080，269 帧，25 fps，10s 渲染时间（以秒为单位）：
| Option | Crystal | Sapphire | Ruby | Opal |
| --- | --- | --- | --- | --- |
| Only Swap | 7.3 | 7.5 | 4.4 | 4.3 |
| Swap+GFPGAN | 10.7 | 11.0 | 9.0 | 9.8 |
| Swap+Codeformer | 12.4 | 13.5 | 11.1 | 11.1 |
| Swap+one word CLIP | 10.4 | 11.2 | 9.1 | 9.3 |
| Swap+Occluder | 7.8 | 7.8 | 4.4 | 4.7 |
| Swap+MouthParser | 13.9 | 12.1 | 5.0 | 4.9 |

Remember, technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of deep fake technology for the betterment of society.



  
