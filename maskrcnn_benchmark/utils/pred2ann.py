from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import os.path as osp
import ipdb
ann_file = 'FLC2019/test/coco_annotations/instances_test_pos.json'
pred_file = 'inference/flc_test_positive_instances_cocostyle/segm.json'

#load GT and Detections
cocoGt = COCO(ann_file)
cocoDt = cocoGt.loadRes(pred_file)
#Eval Detections
cocoEval = COCOeval(cocoGt,cocoDt)
cocoEval.evaluate()

fp_th = 0.8
newann = []
#Find the FPs given a threshold
for key in cocoDt.imgToAnns.keys():
    #ipdb.set_trace()
    ious = cocoEval.ious[(key,1)]
    for i,iou in enumerate(ious):
        max_overlap = max(iou)
        if max_overlap <= fp_th:
            fp = cocoDt.imgToAnns[key][i]
            fp['category_id'] = 2
            fp['width'] = fp['segmentation']['size'][1]
            fp['height'] = fp['segmentation']['size'][0]
            newann.append(fp)

