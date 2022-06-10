# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 11:26:50 2022

@author: assuntaciarlo

real-time T2* log-linear estimation from multi-echo EPI sequence

the implamentation is taken from

https://github.com/jsheunis/fMRwhy/blob/74d264ac06f2dd95bd4b6fb2034939a121b2a64a/fmrwhy/realtime/fmrwhy_realtime_estimateMEparams.m

and adapted to prepare data for Turbo-BrainVoyager preprocessing


Formula derivation for log-linear estimation of T2star and SO:
    
     ---------------------------------------------------------------
      S = S0*exp(-t/T2star) = S0*exp(-t*R2star)
      log(S) = log(S0*exp(-t/T2star)) = log(S0) - t*R2star
     ---------------------------------------------------------------
      [log(S(TE1))]        = [log(S0) - TE1*R2star]
      [log(S(TE2))]          [log(S0) - TE2*R2star]
      [log(S(TE3))]          [log(S0) - TE3*R2star]
                          = [1 - TE1]     [log(S0)]
                            [1 - TE2]  *  [R2star ]
                            [1 - TE3]
      ---------------------------------------------------------------
      Y = X*b (+ e) ==> [log(S(TE))] = [1 - TE]  *  [log(S0)]
                                                    [R2star ]
      ---------------------------------------------------------------
      b_hat = pinv(X)*Y ==> [log(S0)] = pinv([1 - TE]) * [log(S(TE))]
                            [R2star ]
      ---------------------------------------------------------------
      
To obtain the final images per-voxel T2star-weighted combination is applied. 
See Posse et al. and Poser et al. for details.

      ---------------------------------------------------------------
      S = sum[S(TEn).w(TEn)n]
    
      w(TEn)n = [TEn.exp(-TEn/T2*)]/[TEn.exp(-TEn/T2*)]
    
      S = sum[S(TEn)].[TEn.exp(-TEn/T2*)]/[TEn.exp(-TEn/T2*)] 
      ---------------------------------------------------------------


Tested on dicom images from Siemens Prisma_fit scanner (software version syngo MR D13D) and cmrr multi-echo sequence 
(dicom data are assumed to be in Siemens mosaic format)


Reference:
   [1] Heunis, S., Breeuwer, M., Caballero-Gaudes, C., Hellrung, L., Huijbers, W., Jansen, J.F.,
       Lamerichs, R., Zinger, S., Aldenkamp, A.P., 2020. The effects of multi-echo fMRI combination
       and rapid T2*-mapping on offline and real-time BOLD sensitivity. bioRxiv 2020.12.08.416768.
       https://doi.org/10.1016/j.neuroimage.2021.118244

   [2] Heunis, S., Breeuwer, M., Caballero-Gaudes, C., Hellrung, L., Huijbers, W., Jansen, J.F.,
       Lamerichs, R., Zinger, S., Aldenkamp, A.P., 2020. rt-me-fMRI: a task and resting state dataset for real-time,
       multi-echo fMRI methods development and validation. https://doi.org/10.12688/f1000research.29988.1
    
"""

import rtme_utils as utils
import time
import pydicom
import os
from expyriment_stash.extras.expyriment_io_extras import tbvnetworkinterface
import glob
import sys
import logging
from IPython import get_ipython
from tkinter import messagebox

#just for spyder
#get_ipython().run_line_magic('matplotlib', 'qt')

#%%
####################### IMPORT SEQUENCE INFO FROM TBV #########################

# create an instance to access TBV via network plugin
tbv = tbvnetworkinterface.TbvNetworkInterface('localhost',55555)

# get TBV watch folder (where new dicom images must be saved)
tbv_watch_folder =  tbv.get_watch_folder()[0]

# get dicom folder from tbv_watch_folder (assuming /....dicom_folder...../TBVFiles/WatchFolder/)
idx =[i for i, x in enumerate(tbv_watch_folder) if x == '/']
dicom_folder = tbv_watch_folder[:idx[-3]]

# get .tbvj associated with the current tbv project (it must be prj_name.tbvj)
tbv_files_folder = tbv_watch_folder[:idx[-2]]
prj_name = tbv.get_project_name()[0]

#check for the existance of the prj_name.tbvj file
tbvj = glob.glob(os.path.join(tbv_files_folder, prj_name+'.tbvj'))

try:
    assert len(tbvj)!=0, prj_name+".tbvj file not found in the folder '"+tbv_files_folder+\
    "': check that the name of the .tbvj file currently loaded in TBV coincides with the ProjectName in the TimeCourse Tab"   
    tbvj_filename = tbvj[0]     
except AssertionError as msg:
    print(msg)
    messagebox.showerror('File Error', msg)
    sys.exit()
    
# change number of volumes here according to your sequence
num_echos = 4

# read sequence information from .tbvj file
num_vol, ser_num, first_img_num = utils.read_TBV_json(tbvj_filename) 

# CHECK zfill line 114 aline 59 of using.stack_data_all_TE if you are using a different sequence
dcm_fmt = "001_{0}_{1}.dcm" 

#save log file on outputs
logging.basicConfig(filename=tbvj_filename[:-5]+'.log',
            filemode='w',
            level=logging.INFO)

# save also the T2star images, this will increase total computation time
save_t2star = False 

if save_t2star: 
    #make folder to save T2star images
    t2star_dcm_folder = tbv_files_folder+'/T2star_3echos'
    if not(os.path.exists(t2star_dcm_folder)):
        os.mkdir(t2star_dcm_folder)

print('Waiting for dicom images ....')
print('Echo times: ')

############################### START ESTIMATION ##############################

#since we are getting information from TBV and there migth be acquired volumes
#not included in the protocol (e.g., dummy volumes at the beginning of the run), 
#the total number of expected volumes is corrected by the number of the first
#volume TBV is waiting for.

for cur_vol in range(num_vol+first_img_num-1): 
    
    t = time.perf_counter()
    data, echos, dcm = utils.stack_data_all_TE(cur_vol, num_echos, dicom_folder, ser_num, dcm_fmt)
    #print('Time for loading dicom images: {:.3f} sec'.format(time.perf_counter() - t))
    
    #t1 = time.perf_counter()
    T2star, mask_up = utils.T2star_estimation(data, echos)
    #print('Time for T2star estimation: {:.3f} sec'.format(time.perf_counter() - t1))
    
    #t2 = time.perf_counter()
    combined_data = utils.T2star_weighted_comb(data,T2star,echos,mask_up)
    #print('Time for T2star weighted combination: {:.3f} sec'.format(time.perf_counter() - t2))
    
    #t3 = time.perf_counter()
    dcm.PixelData = combined_data.astype('uint16').tobytes()
    dcm.InstanceNumber = cur_vol+1
    pydicom.filewriter.dcmwrite(os.path.join(tbv_watch_folder,dcm_fmt.format(str(ser_num).zfill(6),
    #print('Time for saving dicom image: {:.3f} sec'.format(time.perf_counter() - t3))
        
                                                                             str(cur_vol+1).zfill(6))),dcm, write_like_original=False)
    if save_t2star:
        #save also T2star data
        dcm.PixelData = T2star.astype('uint16').tobytes()
        pydicom.filewriter.dcmwrite(os.path.join(t2star_dcm_folder,dcm_fmt.format(str(ser_num).zfill(6),
                                                                           str(cur_vol+1).zfill(6))),dcm, write_like_original=False)
            
    #print('vol ' + str(cur_vol+1) +' read&write: {:.3f} sec'.format(time.perf_counter() - t) )
    logging.info('vol ' + str(cur_vol+1) +' read&write: {:.3f} sec'.format(time.perf_counter() - t) )
    
    
  
    

