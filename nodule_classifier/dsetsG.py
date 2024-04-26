from functools import lru_cache
import glob
import logging
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from util.utilG import getCacheHandle, unzipped_path, xyz2irc
import SimpleITK as sitk

# log = logging.getLogger('ggggg')
# # log.setLevel(logging.WARN)
# # log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

disk_cache = getCacheHandle('test1')





class CTScan:
    def __init__(self, seriesuid) -> None:
        self.seriesuid = seriesuid
        path_mhdfile = glob.glob(unzipped_path + 'subset*/subset*/{}.mhd'.format(seriesuid))[0]
        ct_img = sitk.ReadImage(path_mhdfile) #contain metadata getters
        ct_img_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        self.ct_img_arr = ct_img_arr
        # #crop HU values to [-1000, 1000]
        # np.clip(ct_img_arr, -1000, 1000, out=ct_img_arr)
        
        self.origin_xyz = np.array(ct_img.GetOrigin())
        self.vxSize_xyz = np.array(ct_img.GetSpacing())
        self.direction_matrix = np.array(ct_img.GetDirection()).reshape(3, 3)
        self.positive_mask = None # holder for positive mask, used by CTScan_seg class
    
    def get_ct_cropped(self, center_xyz, axis_sizes = (32, 48, 48)):
        """
        center_xyz: tuple of 3 floats, center of the chunk in xyz coord
        axis_size: tuple of 3 integers, size of the chunk in each axis. Default is (32, 48, 48)
        """
        center_irc = xyz2irc(np.array(center_xyz), self.origin_xyz, self.vxSize_xyz, self.direction_matrix)
        slices = []
        for idx, axis_size in enumerate(axis_sizes):
            start_idx = int(round(center_irc[idx] - axis_size/2))
            end_idx = int(start_idx + axis_size)

            # if start_idx < 0, set start_idx to 0 and end_idx to axis_size[axis]
            # if end_idx > ct_sizes[axis], set end_idx to axis_size and start_idx to ct_sizes[axis] - axis_size
            ct_sizes = self.ct_img_arr.shape
            
            if start_idx < 0:
                start_idx = 0
                end_idx = axis_size
            if end_idx > ct_sizes[idx]:
                end_idx = ct_sizes[idx]
                start_idx = int(ct_sizes[idx] - axis_size)
            
            slices.append(slice(start_idx, end_idx))
        ct_cropped = self.ct_img_arr[tuple(slices)] # get the cropped chunk of the CT scan
        if self.positive_mask: 
            mask_cropped = self.positive_mask[tuple(slices)] # get the cropped chunk of the positive mask
            return ct_cropped, mask_cropped
        return ct_cropped, None
    
    @staticmethod
    @lru_cache(maxsize=1, typed=True)
    def _get_single_ct_lru_cache(seriesuid): #inner use only
        ct = CTScan(seriesuid)
        return ct
    
    @staticmethod
    @disk_cache.memoize(typed=True)
    def get_ct_cropped_disk_cache(seriesuid, center_xyz, axis_sizes = (32, 48, 48)):
        ct = CTScan._get_single_ct_lru_cache(seriesuid)
        # this is why need lru cache of size 1, 
        # but we shuffled the dataset, size 1 cache is not enough
        # so single ct object cache is only useful during prepcache, the datasets always rely on diskcached ct chunks
        ct_cropped, mask_cropped = ct.get_ct_cropped(center_xyz, axis_sizes)
        #crop HU values to [-1000, 1000]
        np.clip(ct_cropped, -1000, 1000, out=ct_cropped)
        return ct_cropped, mask_cropped


@lru_cache(maxsize=1, typed=True)
def create_df_candidates_info() -> pd.DataFrame:
    """combine data from 2 csv files to generate a dataframe with columns: 'xyzCoord', 'isNodule', 'diameter_mm', with row labels being seriesuid"""
    mhd_list = glob.glob(unzipped_path + 'subset*/subset7/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    df_anno = pd.read_csv(unzipped_path + 'annotations.csv')
    df_cand = pd.read_csv(unzipped_path + 'candidates.csv')

    df_anno_grouped = df_anno.groupby('seriesuid')

    new_df_anno = pd.DataFrame(columns=df_anno.columns)

    new_df_anno = df_anno_grouped.apply(
        lambda group: pd.Series(tuple(group[col].tolist() for col in df_anno.columns[1:])),
        include_groups=False)

    new_df_anno.columns = df_anno.columns[1:]

    new_df_anno['xyzCoord'] = [tuple(zip(x, y, z, d)) for x, y, z, d
                        in zip(new_df_anno['coordX'], new_df_anno['coordY'], new_df_anno['coordZ'], new_df_anno['diameter_mm'])]
    # new_df['xyzCoord'] = new_df['xyzCoord'].apply(lambda x: [(t[:-1], t[-1]) for t in x])

    # new_df.reset_index(inplace=True)
    new_df_anno = new_df_anno.iloc[:, [3,4]]

    # new_df_anno


    df_cand_ondisk = df_cand[df_cand['seriesuid'].isin(presentOnDisk_set)]
    df_cand_grouped = df_cand_ondisk.groupby('seriesuid')

    # new_df_cand_ondisk = pd.DataFrame(columns=df_cand_ondisk.columns)

    new_df_cand_ondisk = df_cand_grouped.apply(
        lambda group: pd.Series(tuple(group[col].tolist() for col in df_cand_ondisk.columns[1:])),
        include_groups=False)

    new_df_cand_ondisk.columns = df_cand_ondisk.columns[1:]

    new_df_cand_ondisk['xyzCoord'] = [tuple(zip(x, y, z)) for x, y, z
                        in zip(new_df_cand_ondisk['coordX'], new_df_cand_ondisk['coordY'],
                                new_df_cand_ondisk['coordZ'])]
    new_df_cand_ondisk = new_df_cand_ondisk.loc[:, ['xyzCoord', 'class']]

    # new_df['xyzCoord'] = new_df['xyzCoord'].apply(lambda x: [(t[:-1], t[-1]) for t in x])
    # new_df.reset_index(inplace=True)
    # new_df_cand_ondisk = new_df_cand_ondisk.iloc[:, [3,4]]
    # new_df_cand_ondisk
    new_df_cand_ondisk['diameter_mm'] = new_df_cand_ondisk['xyzCoord'].apply(lambda x: [0.]*len(x))


    def g_compare_rows(cand_row):
        def delta_mm_is_within_tolerance(candidate_xyz, annotation_xyz, annotation_diameter_mm):
            for i in range(3):
                delta_mm = abs(candidate_xyz[i] - annotation_xyz[i])
                if delta_mm > annotation_diameter_mm / 4:
                    return False
            return True
        if cand_row.name not in new_df_anno.index: #row.name is the seriesuid
            return cand_row
        # print(cand_row)
        anno_row = new_df_anno.loc[cand_row.name]
        for idx, cand_xyz_tup in enumerate(cand_row['xyzCoord']):
            for anno_info_tup in anno_row['xyzCoord']:
                anno_xyz_tup, anno_diameter_mm = anno_info_tup[:-1], anno_info_tup[-1]
                if delta_mm_is_within_tolerance(cand_xyz_tup, anno_xyz_tup, anno_diameter_mm):
                    # row['isNodule_bool'][idx] = True
                    cand_row['diameter_mm'][idx] = anno_diameter_mm
                    break
        # print(anno_row)
        return cand_row
    
    # Apply the compare_rows function to each row of df_cand
    df_cand_new = new_df_cand_ondisk.apply(lambda row: g_compare_rows(row), axis=1)

    df_candidates_info = df_cand_new.explode(['xyzCoord', 'class', 'diameter_mm'])  # type: ignore
    df_candidates_info['class'] = df_candidates_info['class'].astype(int).astype(bool)
    df_candidates_info.rename(columns={'class': 'isNodule'}, inplace=True)
    df_candidates_info['diameter_mm'] = df_candidates_info['diameter_mm'].astype(float)
    # df_candidates_info_reset_index = df_candidates_info.reset_index()
    return df_candidates_info

holder = create_df_candidates_info().reset_index() #anti pattern, should fix this
class LunaDataset(Dataset):
    #custome implementation of a dataset that loads the CT scans and candidate info
    
    
    # df_candidates = create_df_candidates_info().sample(frac=1) 
    """we have to perform stratified split on the dataset, because the dataset is extremely imbalanced"""
    df_candidates = holder
    # num_samples = int(0.7 * len(df_candidates)) # lambda will automatically look for instance attributes
    grouped = df_candidates.groupby('isNodule')
    df_train = grouped.apply(lambda x: x.sample(int(int(0.7 * len(holder)) * len(x) / len(holder))), include_groups=False) \
        .reset_index(drop=False, inplace=False)
    df_val = df_candidates.drop(df_train.index).reset_index(drop=False, inplace=False)
    
    
    # dataloader probably does shallow copy of the object when numworkers > 0
    # so df_candidates must stay outside of __init__ for it to be copied to each worker
    def __init__(self, *, frac=.7, balance=True, augmentation=None) -> None:
        # if not hasattr(LunaDataset, 'df_candidates') or LunaDataset.df_candidates is None:
        #     LunaDataset.df_candidates = create_df_candidates_info() # no copy, beware
        #     LunaDataset.df_candidates = self.df_candidates.sample(frac=1) #shuffle
        # self.frac_split_idx = int(frac * len(self.df_candidates))
        self.balance = balance
        self.positives, self.negatives = self.split_neg_pos(self.df_candidates)
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.df_candidates)
    
    def __getitem__(self, idx):
        if self.balance:
            pos_idx = idx // (self.balance + 1) 
            # every balance + 1 samples, we will have a positive sample

            if idx % (self.balance + 1):
                neg_idx = idx - 1 - pos_idx # adjust the idx 
                neg_idx %= len(self.negatives)
                candidateInfo = self.negatives.iloc[neg_idx]
            else:
                pos_idx %= len(self.positives) #pos_list is small, so need to wraps around, otherwise will overflow
                candidateInfo = self.positives.iloc[pos_idx]
        else: # if balance is fasle, then we don't need to balance the dataset
            # this is for validation set
            candidateInfo = self.df_candidates.iloc[idx]
            
        ct_cropped, _ = CTScan.get_ct_cropped_disk_cache(candidateInfo['seriesuid'], candidateInfo['xyzCoord'])
        ct_cropped = torch.tensor(ct_cropped).unsqueeze(0) # add channel input dimension
        if self.augmentation:
            ct_cropped = self.do_augmentation(ct_cropped) 
        
        isNodule_label = candidateInfo['isNodule']
        # one_hot_encoding_tensor = F.one_hot(labels).to(torch.float32)
        
        # one_hot_encoding_tensor = torch.tensor([0, 1]) if isNodule_label else torch.tensor([1, 0])
        # one_hot_encoding_tensor = one_hot_encoding_tensor.to(torch.long)
        
        one_hot_encoding_tensor = torch.tensor(isNodule_label).to(torch.long)
        return ct_cropped, one_hot_encoding_tensor
    
    def split_neg_pos(self, df_candidates) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (df_candidates[self.df_candidates['isNodule']], 
                df_candidates[~self.df_candidates['isNodule']])
        
    def do_augmentation(self, ct_cropped):
        ct_cropped_tensor = ct_cropped.unsqueeze(0) 
        # add batch dimension, because the affine_grid and grid_sample functions want a batch of images
        # but we only have one image, so we add a batch dimension of 1
        # and we will remove the batch dimension after the augmentation

        transform_t = torch.eye(4)       
        for i in range(3):
            if random.random() > 0.5:
                transform_t[i,i] *= -1 # i,i because we want to be in 3d space

            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = 0.1 * random_float  
            # the last(4th) column of the matrix is for translation, 
            # but the 4th element of the 4th column is always 1 
            # because we are in 3D space, so 4d coord is fixed at 1. 
            # That's why for loop of 3 is enough

            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + 0.2 * random_float
            # i,i because we want to be in 3d space

            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)
            # 2d rotation in x-y plane so we dont break the z axis, because voxels isnt cubic in z axis
            rotation_t = torch.tensor([ # rotation in 2d space
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
            transform_t @= rotation_t

        affine_t = F.affine_grid(
                # the affine_grid function watns a 4x3 matrix for 3d transform, 
                # so we need to remove the last row of the transform matrix
                transform_t[:3].unsqueeze(0).to(torch.float32), 
                list(ct_cropped_tensor.shape),
                align_corners=False,
            )

        augmented_chunk = F.grid_sample(
                ct_cropped_tensor,
                affine_t,
                padding_mode='border',
                align_corners=False,
            )

        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= 20.0

        augmented_chunk += noise_t
        return augmented_chunk.squeeze(0) # remove the batch dimension

# training and validation datasets classes, which share the same parent self.df_candidates, 
# but the shared df_candidates is splitted
class LunaDataset_Train(LunaDataset):
    def __init__(self):
        super().__init__(balance=True, augmentation=True)
        # self.df_candidates = self.df_candidates[:self.frac_split_idx]
        self.df_candidates = self.df_train
        self.positives, self.negatives = self.split_neg_pos(self.df_candidates)
        
        
class LunaDataset_Val(LunaDataset):
    def __init__(self):
        super().__init__(balance=False, augmentation=False)
        # self.df_candidates = self.df_candidates[self.frac_split_idx:]
        self.df_candidates = self.df_val
        # self.positives, self.negatives = self.split_neg_pos(self.df_candidates)
