# results




This repository contains the code implemented for my dissertation at University of Edinburgh. The dissertation title is: On the Applicability of feature matching on embedded devices.

I will provide a description of the files below:

- profile.xlsx - SuperGlue only profiling on Google Cloud

- profile_agx_xavier.xlsx- SuperGlue only profiling on AGX Xavier

- profile_xavier_nx.xlsx - SuperGlue only profiling on Xavier NX

- profile_superpoint_gc.xlsx - SuperGlue and Superpoint profling on Google Cloud

- profile_superpoint_xaviernx.xlsx - SuperGlue and Superpoint profiling on Xavier NX

- profile_superpoint_agx_xavier.xlsx - SuperGlue and Superpoint profiling on AGX Xavier

- profile_compressed_xaviernx.xlsx - SuperGlue and Superpoint compressed model profiling on Xavier NX

- test.py - code used to profile SuperGlue only model on the devices

- match_pairs.py - code used to profile SuperGlue and Superpoint on different devices

- match_pars2.py - code used to profile SuperGlue and Superpoint compressed model on Xavier NX

- superpoint.py - superpoint code used in compressing the model

- superglue.py - superglue code used in compressing the model

- matching.py - class that coordinated everything in the pipeline and was used in compressing the model

Further instructions on configuring the devices:

- Get COCO Dataset Link: wget http://images.cocodataset.org/zips/train2014.zip


- Official SuperPoint + Superglue repo link: https://github.com/magicleap/SuperGluePretrainedNetwork . This was used for profiling SuperGlue and Superpoint combined model.

- SuperGlue repo link: https://github.com/skylook/SuperGlue . This was used to profile SuperGlue only model.

- Install Torch on Nvidia-Jetson: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048

- Configure TensorRT: https://docs.donkeycar.com/guide/robot_sbc/tensorrt_jetson_nano/



