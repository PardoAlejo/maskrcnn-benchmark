from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import os.path as osp
import json

def pred2hard(ann_file, pred_file, output, fp_th=0.8):
    
    #load GT and Detections
    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(pred_file)
    #Eval Detections
    cocoEval = COCOeval(cocoGt,cocoDt)
    cocoEval.evaluate()

    #number of instances
    num_train_inst = len(cocoGt.catToImgs[1])

    # Load previous file
    new_anns = json.load(open(ann_file))

    #Find the FPs given a threshold
    for key in cocoDt.imgToAnns.keys():
        #ipdb.set_trace()
        ious = cocoEval.ious[(key,1)]
        for i,iou in enumerate(ious):
            max_overlap = max(iou)
            if max_overlap <= fp_th:
                fp = cocoDt.imgToAnns[key][i]
                fp['category_id'] = 2
                fp['id'] += num_train_inst
                fp['width'] = fp['segmentation']['size'][1]
                fp['height'] = fp['segmentation']['size'][0]
                new_anns['annotations'].append(fp)

    new_anns['categories'].append({'id':2, 'name': 'Three Leaf Clover', 'supercategory': 'Clover'})
    with open(osp.join(output, 'instances_test_hard_negatives.json')) as f:
        json.dump(new_anns, f)
if __name__ == "__main__":
    ann_file = 'examples/instances_test_pos.json'
    pred_file = 'examples/segm.json'
    output = 'examples/'
    pred2hard(ann_file, pred_file, output)
    