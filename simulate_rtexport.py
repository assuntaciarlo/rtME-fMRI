# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:21:30 2022

@author: assun
"""

import os,glob
from expyriment_stash.extras.expyriment_io_extras import tbvnetworkinterface
import glob
import sys
from time import sleep
from shutil import copyfile

#folder 
orig_dicom_dir = 'E:/DataAachen/RSA_cmrr/RSA_cmrr/dcm_orig'

# create an instance to access TBV via network plugin
tbv = tbvnetworkinterface.TbvNetworkInterface('localhost',55555)

# get TBV watch folder (where new dicom images must be saved)
tbv_watch_folder =  tbv.get_watch_folder()[0]

# real-time export folder
# get real-time export folder from tbv_watch_folder (assuming /....dicom_folder...../TBVFiles/WatchFolder/)
idx =[i for i, x in enumerate(tbv_watch_folder) if x == '/']
dicom_folder = tbv_watch_folder[:idx[-3]]


dcm = glob.glob(orig_dicom_dir+'/*.dcm')

for i in range(len(dcm)):
    
    filename = os.path.basename(dcm[i])
    copyfile(dcm[i], os.path.join(dicom_folder, filename))
    sleep(0.01)

