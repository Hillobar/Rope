# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import glob
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm


from .retinaface import RetinaFace
from .arcface_onnx import ArcFaceONNX
from .common import Face

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name, root='~/.insightface', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(4)
        self.models = {}

        session = onnxruntime.InferenceSession('.\models\det_10g.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.models['detection'] = RetinaFace('.\models\det_10g.onnx', session=session)

        session = onnxruntime.InferenceSession('.\models\w600k_r50.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.models['recognition'] = ArcFaceONNX('.\models\w600k_r50.onnx', session=session)       

        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        # print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret



