from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
import xlsxwriter
# Datasets
from datasets.sift_dataset import SIFTDataset
from datasets.superpoint_dataset import SuperPointDataset
import os
import torch.multiprocessing
from tqdm import tqdm

import torch
import torchvision
import torchprof

# torch.backends.cudnn.benchmark = True

# from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified)

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.matchingForTraining import MatchingForTraining

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# torch.multiprocessing.set_start_method("spawn")
# torch.cuda.set_device(0)

# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass


parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the matches and dump the plots')
parser.add_argument(
    '--eval', action='store_true',
    help='Perform the evaluation'
            ' (requires ground truth pose and intrinsics)')

parser.add_argument(
    '--detector', choices={'superpoint', 'sift'}, default='superpoint',
    help='Keypoint detector')
parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. If two numbers, '
            'resize to the exact dimensions, if one number, resize the max '
            'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--cache', action='store_true',
    help='Skip the pair if output .npz files are already found')
parser.add_argument(
    '--show_keypoints', action='store_true',
    help='Plot the keypoints in addition to the matches')
parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization based on OpenCV instead of Matplotlib')
parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')

parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
parser.add_argument(
    '--eval_pairs_list', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
    help='Path to the list of image pairs for evaluation')
parser.add_argument(
    '--shuffle', action='store_true',
    help='Shuffle ordering of pairs before processing')
parser.add_argument(
    '--max_length', type=int, default=-1,
    help='Maximum number of pairs to evaluate')

parser.add_argument(
    '--eval_input_dir', type=str, default='assets/scannet_sample_images/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--eval_output_dir', type=str, default='dump_match_pairs/',
    help='Path to the directory in which the .npz results and optional,'
            'visualizations are written')
parser.add_argument(
    '--learning_rate', type=int, default=0.0001,
    help='Learning rate')

    
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')
parser.add_argument(
    '--train_path', type=str, default='assets/freiburg_sequence', # MSCOCO2014_yingxin
    help='Path to the directory of training imgs.')
# parser.add_argument(
#     '--nfeatures', type=int, default=1024,
#     help='Number of feature points to be extracted initially, in each img.')
parser.add_argument(
    '--epoch', type=int, default=20,
    help='Number of epoches')



if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'



    # store viz results
    eval_output_dir = Path(opt.eval_output_dir)
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write visualization images to',
        'directory \"{}\"'.format(eval_output_dir))

    # detector_factory = {
    #     'superpoint': SuperPointDataset,
    #     'sift': SIFTDataset,
    # }
    detector_dims = {
        'superpoint': 256,
        'sift': 128,
    }

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
            'descriptor_dim': detector_dims[opt.detector],
        }
    }
    config2= {

        'superpoint': {

        'nms_radius': opt.nms_radius,

        'keypoint_threshold': opt.keypoint_threshold,

        'max_keypoints': opt.max_keypoints,

        },

        'superglue': {

        'weights': 'indoor',

        'sinkhorn_iterations': opt.sinkhorn_iterations,

        'match_threshold': opt.match_threshold,

        'descriptor_dim': detector_dims[opt.detector],

        }

        }


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set data loader
    if opt.detector == 'superpoint':
        train_set = SuperPointDataset(opt.train_path, device=device, superpoint_config=config.get('superpoint', {}))
    elif opt.detector == 'sift':
        train_set = SIFTDataset(opt.train_path, nfeatures=opt.max_keypoints)
    else:
        RuntimeError('Error detector : {}'.format(opt.detector))

    test_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True)

    # superpoint = SuperPoint(config.get('superpoint', {}))
    #superglue = SuperGlue(config.get('superglue', {}))
    
    #superglue.eval()[19:23] WEN Xiangyu
    superglue = SuperGlue(config2.get('superglue', {}))



    if torch.cuda.is_available():
        # superpoint.cuda()
        superglue.cuda()
    else:
        print("### CUDA not available ###")
    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)

    mean_loss = []
    with torch.no_grad():
        with torchprof.Profile(superglue, use_cuda=True, profile_memory=True) as prof:
            for i, pred in enumerate(test_loader):
                for k in pred:
                    if k != 'file_name' and k!='image0' and k!='image1':
                        if type(pred[k]) == torch.Tensor:
                            pred[k] = Variable(pred[k].cuda())
                        else:
                            pred[k] = Variable(torch.stack(pred[k]).cuda())
                if i==4000:
                    break
                    
                data = superglue(pred)
                #print(i)

            
        print(prof.display(show_events=False))

        result = ""
        result += prof.display()
        trace, event_lists_dict = prof.raw()
        run_time = 1.4 #set the time threshold(ms) to filter the layers running longer than threshold
        result += "\n\nThe layers cost longer than %fs: \n" % run_time
        for i in range(0,len(trace)):
            if len(event_lists_dict[trace[i].path]):
                if event_lists_dict[trace[i].path][0].self_cpu_time_total>run_time*1000000:
                    #print(trace[i].path)
                    #print(dir(event_lists_dict[trace[i]]))
                    result += "%s\t\t\tCost time in CPU:%fs\n" % (str(trace[i].path),(event_lists_dict[trace[i].path][0].self_cpu_time_total/1000000))
                    #result += "%s\n" % str(event_lists_dict[trace[i].path][0]) #show the details of these costy modules
            else:
                i+=1

        print(result)
        #workbook = xlsxwriter.Workbook('profile.xlsx')
        #worksheet = workbook.add_worksheet()
        #worksheet.write(0,0,'Profiling')
        #workbook.close()
        excel = open("profile.xlsx", "w").write(result)
        

