# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:20:36 2022

Rt-me utils

@author: assuntaciarlo

"""

import numpy as np
import pydicom
import time
from numpy.linalg import pinv
import os, glob
import json
from time import sleep

def read_TBV_json(jsonfile):
    
    '''
    derive sequence information from .tbvj file
    
    Inputs:
    ---------
    
        jsonfile: string
            .tbvj filename
            
    Outputs:
    ---------
    
        num_vol: int
            number of exspected volumes
        ser_num: int 
            dicom series number 
        first_img_num: int
            number of the first volume to be processed by TBV
    
    '''
    tbv_data = json.load(open(jsonfile))
    num_vol = tbv_data['TimeCourseInfo']['NrOfVolumes']
    ser_num = tbv_data['DataFormatInfo']['DicomFirstVolumeNr']
    first_img_num = tbv_data['DataFormatInfo']['DicomFirstImageNr']
    
    return int(num_vol), int(ser_num), int(first_img_num)

def stack_data_all_TE(cur_vol, num_echos, dicom_folder, ser_num, dcm_fmt):
    
    '''
    Collect dcm for all echoes at vol = cur_vol
    Dicom images are written sequencially in the dicom folder
    
    Inputs:
    ---------
    
        cur_vol: int
            current volume number
        num_echos: int
            number of echos
        dicom_folder: string
            real-time export folder
        ser_num: int
            dicom series number
        dcm_fmt: string
            dicom name format        
    
    Outputs:
    ---------
    
        data: np.array [x,y,z,num_echos]
            stacked data of the current volume for all echos
        echos: np.array [1,num_echos]
            TEs
        dcm: pydicom object
            object containing dicom header and data
    
    '''
    
    data = [] #variable to stack data of all TEs
    echos = []
    for i in range(num_echos):
        dcm = [] #istance for dicom path 
        is_fully_written = False
        size = -1
        cur_dcm = cur_vol*num_echos+i+1
        
        while len(dcm)==0 and not(is_fully_written): # waiting for dicom
            
            dcm = glob.glob(os.path.join(dicom_folder, dcm_fmt.format(str(ser_num).zfill(6),
                                                                      str(cur_dcm).zfill(6)) ))
            
            if len(dcm)!=0: # if the file exists check if it's fully written
               if os.path.getsize(dcm[0]) != size:
                   size = os.path.getsize(dcm[0])
               else:
                   is_fully_written = True          
            sleep(0.001)
            
        dcm = pydicom.dcmread(dcm[0])
        data.append(dcm.pixel_array) 
         
        if cur_vol == 0:
            print(dcm.EchoTime)
            
        echos.append(dcm.EchoTime.real)
        
    return np.array(data), np.array(echos), dcm


def T2star_estimation(data, echos):
    
    '''
    T2* log-linear estimation
    
    data masking is skipped
    look here (https://github.com/ME-ICA/tedana/blob/main/tedana/utils.py)
    
     Y = X*b (+ e) ==> [log(S(TE))] = [1 - TE]  *  [log(S0)]
                                                    [R2star]
    ---------------------------------------------------------------
    
     b_hat = pinv(X)*Y ==> [log(S0)] = pinv([1 - TE]) * [log(S(TE))]
                            [R2star ]
    
    Inputs:
    ---------
        data: np.array [num_echos,x,y,z]
            stacked data of the current volume for all echos 
        echos: np.array [1,num_echos]
            TEs
            
    Outputs:
    ---------
        T2star: np.array [mosaic_x, mosaic_y]
            T2 star estimated map in mosaic format
        
        mask_up: np.array [mosaic_x, mosaic_y]
            mask of the pixels with an estimated T2 star higher than 500
        
       mask_down: np.array [mosaic_x, mosaic_y]
            mask of the pixels with an estimated T2 star lower than 500

    '''
    num_echos = data.shape[0]
    #reshape data as 2D matrix [time x voxels]
    mosaic_dim = data.shape[1:]
    Y = data.reshape(data.shape[0],mosaic_dim[0]*mosaic_dim[1])
    
    #search for value <=0
    Y1 = np.maximum(Y,1e-16) # replace zero with very small value to compute the logatitm
    
    # define X matrix
    X = np.hstack((np.ones([num_echos,1]),-echos.reshape(-1,1)))
    b_hat = (pinv(X)).dot(np.log(Y1)) # [log(S0);R2star]
    
    T2star = np.zeros(b_hat[1,:].shape)
    nozero_idx = np.where(b_hat[1,:]!=0)
    T2star[nozero_idx] = abs(1/b_hat[1,nozero_idx])
    
    # thresholding T2star map
    T2star_thresh_max = 500 # same as tedana.
    
    
    #just for testing return a mask of voxels > T2star_thresh_max and 
    mask_up = np.zeros(T2star.shape)
    mask_up[np.where(T2star>=T2star_thresh_max)] = 1 
    
    T2star[np.where(T2star>T2star_thresh_max)] = T2star_thresh_max #as tedana
    
    
    T2star = T2star.reshape(mosaic_dim[0], mosaic_dim[1])
    mask_up = mask_up.reshape(mosaic_dim[0], mosaic_dim[1])
    
 
    
    return T2star, mask_up 
    
    
 
def T2star_weighted_comb(data,T2star,echos,mask_up):
    
    '''
    T2star-fit weighted combination 
    
    Per-voxel T2star-weighted combination. This 3D T2* image could
    be calculated in various ways (e.g. T2star and S0 estimated per voxel
    from time-series average of multiple echoes; or T2star and S0 estimated
    in real-time). See Posse et al. 1999 and Poser et al. 2018 for details.
    
    --------------------------------------------------------------------------
    
    S = sum[S(TEn) x w(TEn)n]
    
    w(TEn)n = [TEn x exp(-TEn/T2*)]/[TEn x exp(-TEn/T2*)]
    
    thus, S = sum[S(TEn) x [TEn x exp(-TEn/T2*)]/[TEn x exp(-TEn/T2*)]
    
    --------------------------------------------------------------------------
    
    Inputs:
    ---------
        data: np.array [num_echos,x,y,z]
            stacked data of the current volume for all echos 
        T2star: np.array [mosaic_x, mosaic_y]
            T2 star estimated map in mosaic format
        echos: np.array [1,num_echos]
            TEs  
        mask_down: np.array [mosaic_x, mosaic_y]
            mask of the pixels with an estimated T2 star lower than 500
    Outputs:
    ---------
        combined data: np.array [mosaic_x, mosaic_y]
            T2star weighted combination of the multi-echos images
    
    '''
    #use mask to avoid nans in the final image
    nozero_mask = np.zeros(T2star.shape).astype(bool)
    nozero_mask[np.where(T2star !=0 )] = 1
   
    weights = np.zeros(data.shape)
    for i in range(len(echos)):
        #weights.append(echos[i]*np.exp(-echos[i]/T2star)) 
        weights[i,nozero_mask] = echos[i]*np.exp(-echos[i]/T2star[nozero_mask]) 
    weights = np.array(weights) 
    weighted_data = data*weights
    sum_weights = np.sum(weights,axis=0)
    sum_weighted_data = np.sum(weighted_data,axis=0)
    
    mask_nozero_w = np.zeros(sum_weights.shape).astype(bool)
    mask_nozero_w[np.where(sum_weights !=0 )] = 1
    combined_data = np.zeros(T2star.shape)
    combined_data[mask_nozero_w] = sum_weighted_data[mask_nozero_w]/sum_weights[mask_nozero_w]
    
    
    return combined_data