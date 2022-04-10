# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch

from .superpoint import SuperPoint
from .superglue import SuperGlue
from torch.autograd import Variable


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}
        pred['image0'] = torch.stack([data[0,0,:,:,:]])
        pred['image1'] = torch.stack([data[0,1,:,:,:]])
        data = {}
        data['image0'] = pred['image0']
        data['image1'] = pred['image1']

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in pred:
            #print(pred['image0'])
            pred0 = self.superpoint({'image': pred['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
            print('Keypoints0')
        if 'keypoints1' not in pred:
            #pred1 = self.superpoint({'image': pred['image1']})
            pred1 = pred0
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
            print('keypoints1')
        


        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}
        #print(data['keypoints0'])
        for k in data:
            if isinstance(data[k], (list, tuple)) and len(data[k]) > 0:
                #print(data[k])
                print(k)
                if k!='keypoints0' and k !='keypoints1' and k!='scores0' and k !='scores1' and k!='descriptors0' and k!='descriptors1':
                    data[k] = torch.stack(data[k])
                else:
                    data[k] = torch.as_tensor([data[k][0].tolist()]).cuda()
        # Perform the matching
        pred = {**pred, **self.superglue(data)}
        #print('Done')
        #print(type(pred['scores0'][0]))
        #print(type(pred['scores0'][0]))
        return pred['matches0']
