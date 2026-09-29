# -*- coding: utf-8 -*-
"""中文注释文案（单独文件，方便以后修改）。

本文件只存放"悬停注释"（tooltip）的中文文案：每个设置项的 1-2 句说明、
以及选择类控件每个选项按钮的单独说明。界面翻译见 zh_text.py。
"""

# 每个参数的中文悬停说明（1-2 句）
ZH_INFO: dict[str, str] = {
    "StartRope": "启动换脸主流程，配合目标视频和源人脸使用。",
    "OutputFolder": "选择换脸结果（视频/图片）的保存目录。",
    "EmbedFile": "选择嵌入文件（.dat 人脸特征库）。",
    "ClearVramButton": "立即释放显存缓存；显存吃紧或变卡时点一下。",
    "BenchmarkButton": "以极限速度跑完当前视频，输出每帧耗时与 FPS 报告，用于对比性能。",
    "BenchmarkHeadlessButton": "不渲染界面的性能测试，结果更接近硬件真实吞吐。",
    "LoadTVideos": "选择存放目标视频的文件夹，批量导入待换脸的视频。",
    "LoadSFaces": "选择存放源人脸图片的文件夹，批量导入用于换脸的身份照片。",
    "Audio": "播放时是否输出视频原声。需在点击播放前开启。",
    "MaskView": "预览时是否叠加显示遮罩区域。",
    "SaveImageButton": "把当前预览帧保存为图片。",
    "AutoSwapButton": "自动换脸模式开关。",
    "TLBeginning": "跳转到视频开头。",
    "TLLeft": "向后退 30 帧。",
    "Record": "开始/停止录制当前换脸结果。",
    "Play": "播放 / 暂停。",
    "TLRight": "向前进 30 帧。",
    "AddMarkerButton": "在当前帧添加一个标记点。",
    "DelMarkerButton": "删除当前帧的标记点。",
    "PrevMarkerButton": "跳到上一个标记点。",
    "NextMarkerButton": "跳到下一个标记点。",
    "FindFaces": "在预览帧中检测并列出所有人脸。",
    "ClearFaces": "清除当前帧已找到的人脸列表。",
    "SwapFaces": "换脸总开关：点亮后播放时才执行换脸。",
    "DelEmbed": "删除当前选中的嵌入（人脸特征）。",
    "SaveParamsButton": "把当前全部参数保存到配置文件。",
    "LoadParamsButton": "从配置文件加载已保存的参数。",
    "DefaultParamsButton": "恢复全部默认参数。",
    "FindSimilarFacesButton": "在当前帧中查找与选中人脸相似的人脸。",
    "CapturePlayButton": "开始/停止屏幕捕捉换脸（取景框内区域）。",
    "CaptureReopenButton": "重新打开屏幕捕捉的取景框。",
    "PreviewModeTextSel": "选择预览显示模式（原画面/换脸结果等）。",
    "RestorerSwitch": "启用人脸修复增强，提升输出清晰度。",
    "RestorerTypeTextSel": "选择修复模型：GFPGAN、GPEN 或 CodeFormer。",
    "RestorerDetTypeTextSel": "修复前人脸对齐所用的检测方式。",
    "RestorerSlider": "修复结果与原图的混合强度。",
    "ThresholdSlider": "人脸匹配的最低相似度，高于此值才执行换脸。",
    "OrientSwitch": "启用人脸方向校正，改善侧脸/歪脸。",
    "OrientAutoSwitch": "自动探测人脸倾斜角度，无需手动。",
    "OrientSlider": "手动指定人脸旋转角度。",
    "StrengthSwitch": "启用换脸强度调节。",
    "StrengthSlider": "换脸效果强弱：越高越像源脸。",
    "ColorMatchSwitch": "用 LAB 颜色匹配，让换脸后的肤色更自然。",
    "LikenessSlider": "换脸结果与源脸的相似程度。",
    "EmbExtrapSlider": "区分度：越高越像源脸，越低越像目标脸。",
    "HighFidelitySwitch": "两遍高保真处理，画质更高但更慢。",
    "HighFidelityAlphaSlider": "高保真校正结果的混合比例。",
    "HighFidelityModeTextSel": "高保真处理模式。",
    "HFRefineButton": "重新构建高保真细化缓存。",
    "HFClearCacheButton": "清除高保真缓存，下次重新生成。",
    "BorderTopSlider": "遮罩顶部留白，避免额头之外区域被替换。",
    "BorderSidesSlider": "遮罩左右留白，控制替换范围。",
    "BorderBottomSlider": "遮罩底部留白。",
    "BorderBlurSlider": "遮罩边缘羽化程度，越大过渡越柔和。",
    "BlendSlider": "整体遮罩的融合强度。",
    "DiffSwitch": "差分模式：只处理画面中变化的区域。",
    "DiffSlider": "差分处理的灵敏度。",
    "OccluderSwitch": "启用遮挡检测（眼镜、头发等）。",
    "OccluderSlider": "遮挡检测范围的大小。",
    "DFLXSegSwitch": "启用 XSeg 遮罩，处理遮挡和头发干扰。",
    "DFLXSegSizeSlider": "XSeg 遮罩的工作尺寸。",
    "DFLXSegBlurSlider": "XSeg 遮罩边缘模糊。",
    "FaceParserSwitch": "启用人脸语义解析，可按区域精细控制。",
    "FaceParserSlider": "背景区域的处理比例。",
    "MouthParserSlider": "嘴部区域的处理比例。",
    "ColorSwitch": "启用手动颜色调整。",
    "ColorRedSlider": "红色通道增益。",
    "ColorGreenSlider": "绿色通道增益。",
    "ColorBlueSlider": "蓝色通道增益。",
    "ColorGammaSlider": "伽马校正，调整整体亮度曲线。",
    "ColorContrastSlider": "对比度调整。",
    "ColorSaturationSlider": "饱和度调整。",
    "FaceAdjSwitch": "启用输入人脸的关键点/缩放调整。",
    "KPSXSlider": "关键点横向偏移。",
    "KPSYSlider": "关键点纵向偏移。",
    "KPSScaleSlider": "关键点整体缩放。",
    "FaceScaleSlider": "人脸整体缩放。",
    "FindSimilarThresholdSlider": "相似人脸查找的匹配阈值。",
    "ThreadsSlider": "并行处理线程数，越高越快但更吃显存。",
    "ModelSessionsTextSel": "每个模型创建的会话数（每个线程独立会话）。",
    "DetectTypeTextSel": "人脸检测模型：Retinaface 或 SCRDF。",
    "DetectInputSizeTextSel": "检测输入分辨率：640 最稳，越小越快但小脸易漏。",
    "DetectScoreSlider": "人脸检测的置信度阈值。",
    "RecordTypeTextSel": "录制输出的封装格式。",
    "VideoQualSlider": "输出视频的编码质量。",
    "MergeTextSel": "换脸融合的数学方式：把选中的多张源人脸 512 维嵌入合成一个身份向量。\n"
        "Mean（默认）：逐元素平均，最快最稳，但一张坏脸会拉偏结果。\n"
        "Median：逐轴取中位数，不怕单张坏脸，但是跨脸拼出的合成向量，身份可能不连贯。\n"
        "Sph：归一化后取球面质心，贴合 ArcFace 球面几何，通常比 Mean 小幅提升。\n"
        "Geo：几何中位数（Weiszfeld 迭代），最抗离群、结果是真实点，但计算最慢。\n"
        "Qual：按嵌入模长加权平均，源脸质量参差时最合适，依赖“模长≈质量”假设。",
    "SwapperTypeTextSel": "换脸输出分辨率：128 是默认模型最快最稳；256/512 需要额外的模型文件，且更吃显存。",
    "CaptureFPSSlider": "屏幕捕捉的帧率。",
    "CaptureBorderSlider": "取景框边框的粗细。",
    "CaptureLockRegionSwitch": "锁定捕捉区域，防止误拖动。",
    "CaptureAlwaysOnTopSwitch": "取景框窗口保持置顶。",
    "LanguageSelect": "界面语言：English 纯英文；中文 全中文界面；English + 中文注释 英文界面但悬停有中文说明。切换后重启生效。",
    "TipsSwitch": "悬停注释总开关：关闭后不再显示鼠标悬停时弹出的中文注释。切换后重启生效。",
}

# 选择类设置项：每个选项按钮/下拉项单独的中文悬停说明（键 = 选项值）
ZH_MODE_INFO: dict[str, dict[str, str]] = {
    "MergeTextSel": {
        "Mean": "优点：逐元素平均，最快最稳、无额外开销（默认）。\n缺点：一张模糊/角度差的坏脸会拉偏整体结果，所有脸权重相同。",
        "Median": "优点：每个维度取中位数，单张坏脸的极端值会被剔除，比较抗噪。\n缺点：结果是从不同脸的各个维度拼出来的合成向量，不是任一真实脸，身份可能不连贯。",
        "Sph": "优点：每张脸先 L2 归一化，取均值后再归一化，贴合 ArcFace 训练所用的单位球面几何，通常比 Mean 小幅提升。\n缺点：和 Mean 一样对离群脸不够鲁棒。",
        "Geo": "优点：几何中位数（Weiszfeld 迭代），求到所有输入脸 L2 距离总和最小的真实点，最抗离群。\n缺点：迭代计算最慢，效果依赖收敛精度。",
        "Qual": "优点：按每张脸的嵌入模长加权平均（MagFace 文献以模长作为人脸质量代理），源脸质量参差时效果最好。\n缺点：依赖“模长≈质量”假设，个别低质高模长的脸可能被误判。",
    },
    "SwapperTypeTextSel": {
        "128": "优点：内置默认模型 inswapper_128，最快最稳、显存占用最低（默认）。\n缺点：输出分辨率低，特写/大屏看细节略糊。",
        "256": "优点：细节明显优于 128。\n缺点：需要额外模型文件 inswapper_256_phase1.onnx（官方模型包不含），更吃显存，首次运行要构建 TRT 引擎较慢。",
        "512": "优点：细节最丰富，特写效果最好。\n缺点：需要额外模型文件 inswapper_512_level2.onnx（官方模型包不含），最吃显存（12GB 显存较紧张），首次构建引擎最慢。",
    },
    "RestorerTypeTextSel": {
        "GFPGAN": "优点：通用人脸修复，效果均衡、最常用（默认）。\n缺点：速度中等。",
        "CF": "优点：CodeFormer，对低质量/压缩严重的脸修复更好。\n缺点：可能轻微改变面部特征，速度较慢。",
        "GPEN256": "优点：GPEN 256 版，速度快、细节自然。\n缺点：分辨率上限 256。",
        "GPEN512": "优点：GPEN 512 版，清晰度更高。\n缺点：更吃显存，速度更慢。",
    },
    "RestorerDetTypeTextSel": {
        "Original": "沿用检测阶段得到的原始对齐。\n缺点：极端角度下修复可能错位。",
        "Blend": "优点：原图与参考人脸按比例混合对齐（默认），兼顾自然与稳定。",
        "Reference": "优点：严格按参考人脸对齐，角度一致时最稳。\n缺点：目标与参考角度差异大时可能失真。",
    },
    "DetectTypeTextSel": {
        "Retinaface": "优点：det_10g 模型，精度高，对侧脸、大角度更稳（默认）。\n缺点：速度较慢。",
        "SCRDF": "优点：scrfd_2.5g 模型，速度更快、更轻量。\n缺点：极小或极模糊的人脸检出率略低。",
    },
    "DetectInputSizeTextSel": {
        "320": "优点：最快，适合低分辨率视频/性能优先。\n缺点：小脸容易漏检。",
        "416": "速度与精度较为平衡。",
        "480": "精度更好，速度尚可。",
        "640": "优点：最稳、精度最高（默认）。\n缺点：最慢、最吃显存。",
    },
    "ModelSessionsTextSel": {
        "Shared": "优点：所有线程共用一个模型会话，省显存（默认）。\n缺点：多线程并发时可能有排队。",
        "Per-Thread": "优点：每个线程独立会话，并行度更高。\n缺点：更吃显存。",
    },
    "RecordTypeTextSel": {
        "FFMPEG": "优点：FFMPEG 编码，画质受 Video Quality 滑块控制，文件大小可控（默认）。\n缺点：速度稍慢。",
        "OPENCV": "优点：OpenCV 编码，速度更快。\n缺点：无画质参数，文件体积较大。",
    },
    "PreviewModeTextSel": {
        "Video": "预览视频画面（默认）。",
        "Capture": "预览屏幕捕捉取景框内的画面。",
    },
    "HighFidelityModeTextSel": {
        "Cached": "优点：先把每帧结果缓存，播放时直接复用，速度快（默认）。\n缺点：首次构建缓存耗时。",
        "Per-frame": "优点：无需缓存，实时逐帧计算。\n缺点：每帧都重算，速度较慢。",
    },
    "LanguageSelect": {
        "English": "纯英文界面。",
        "中文": "全中文界面。",
        "English + 中文注释": "英文界面，悬停时显示中文注释。",
    },
}

# 插件自带控件在英文界面下的悬停说明
EN_INFO: dict[str, str] = {
    "LanguageSelect": "UI language: English, 中文 (full Chinese UI), or English + 中文注释 (English UI with Chinese hover notes). Takes effect after restart.",
    "TipsSwitch": "Master switch for hover annotations: OFF hides the Chinese tooltips shown on hover. Takes effect after restart.",
}
