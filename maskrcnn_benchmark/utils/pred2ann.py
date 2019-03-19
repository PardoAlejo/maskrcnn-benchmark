from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools.mask import decode
from pycococreatortools.pycococreatortools import binary_mask_to_polygon
import os
import os.path as osp
import json
import argparse
import ipdb
from tqdm import tqdm

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
    for key in tqdm(cocoDt.imgToAnns.keys()):
        # ipdb.set_trace()
        ious = cocoEval.ious[(key,1)]
        for i,iou in enumerate(ious):
            max_overlap = max(iou)
            if max_overlap <= fp_th:
                num_fp += 1
                fp = cocoDt.imgToAnns[key][i]
                fp['width'] = fp['segmentation']['size'][1]
                fp['height'] = fp['segmentation']['size'][0]
                fp_segm = decode(fp['segmentation'])
                fp['segmentation'] = binary_mask_to_polygon(fp_segm, tolerance=2)
                # ipdb.set_trace()
                fp.pop('score')
                fp['category_id'] = 2
                fp['id'] += num_train_inst
                fp['area'] = int(fp['area'])
                fp['bbox'] = fp['bbox'].tolist()
                new_anns['annotations'].append(fp)
    print('{} FPs found'.format(num_fp))
    # ipdb.set_trace()
    new_anns['categories'].append({'id':2, 'name': 'Three Leaf Clover', 'supercategory': 'Clover'})
    with open(output, 'w') as f:
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
    
