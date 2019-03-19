from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import os.path as osp
import json
import argparse

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
    print('Finding FP...')
    num_fp = 0
    for key in cocoDt.imgToAnns.keys():
        #ipdb.set_trace()
        ious = cocoEval.ious[(key,1)]
        for i,iou in enumerate(ious):
            max_overlap = max(iou)
            if max_overlap <= fp_th:
                num_fp += 1
                fp = cocoDt.imgToAnns[key][i]
                fp['category_id'] = 2
                fp['id'] += num_train_inst
                fp['width'] = fp['segmentation']['size'][1]
                fp['height'] = fp['segmentation']['size'][0]
                new_anns['annotations'].append(fp)
    print('{} FPs found'.format(num_fp))
    
    new_anns['categories'].append({'id':2, 'name': 'Three Leaf Clover', 'supercategory': 'Clover'})
    with open(output) as f:
        json.dump(new_anns, f)
    
    print('Hard negatives annotation file saved in {}'.format(output))
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predictions to hard negatives")
    parser.add_argument(
        "--ann_file",
        default="",
        help="path to ann file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--pred_file",
        default="",
        help="path to pred file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--out_file",
        default="",
        help="name and path of out file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--th",
        default=0.8,
        help="FP threshold",
        required=False,
        type=float,
    )

    args = parser.parse_args()
    print('Function called with args {}'.format(args))
    #ann_file = 'examples/instances_test_pos.json'
    #pred_file = 'examples/segm.json'
    #output = 'examples/instances_test_hard_negatives.json'
    pred2hard(args.ann_file, args.pred_file, args.out_file)
    
