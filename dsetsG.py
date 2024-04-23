from functools import lru_cache
import glob
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from test.util.util import xyz2irc
from utilG import getCacheHandle, unzipped_path
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
        #crop HU values to [-1000, 1000]
        np.clip(ct_img_arr, -1000, 1000, out=ct_img_arr)
        
        self.origin_xyz = np.array(ct_img.GetOrigin())
        self.vxSize_xyz = np.array(ct_img.GetSpacing())
        self.direction_matrix = np.array(ct_img.GetDirection()).reshape(3, 3)
    
    def get_candidate_croppedChunk_inVoxelCoord(self, center_xyz, axis_sizes = (32, 48, 48)):
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
        ct_cropped = self.ct_img_arr[tuple(slices)]
        return ct_cropped

@lru_cache(maxsize=1, typed=True)
def get_single_ct_lru_cache(seriesuid):
    ct = CTScan(seriesuid)
    return ct

@disk_cache.memoize(typed=True)
def get_ct_cropped_disk_cache(seriesuid, center_xyz, axis_sizes = (32, 48, 48)):
    ct = get_single_ct_lru_cache(seriesuid)
    # this is why need lru cache of size 1, 
    # but we shuffled the dataset, size 1 cache is not enough
    # so single ct object cache is only useful during prepcache, the datasets always rely on diskcached ct chunks
    return ct.get_candidate_croppedChunk_inVoxelCoord(center_xyz, axis_sizes)

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
    return df_candidates_info


class LunaDataset(Dataset):
    #custome implementation of a dataset that loads the CT scans and candidate info
    df_candidates = create_df_candidates_info().sample(frac=1) 
    # dataloader probably does shallow copy of the object when numworkers > 0
    # so df_candidates must stay outside of __init__ for it to be copied to each worker
    def __init__(self, frac=.7):
        # if not hasattr(LunaDataset, 'df_candidates') or LunaDataset.df_candidates is None:
        #     LunaDataset.df_candidates = create_df_candidates_info() # no copy, beware
        #     LunaDataset.df_candidates = self.df_candidates.sample(frac=1) #shuffle
        self.frac_split_idx = int(frac * len(self.df_candidates))
    
    def __len__(self):
        return len(self.df_candidates)
    
    def __getitem__(self, idx):
        # log.debug(self.df_candidates)
        candidateInfo = self.df_candidates.iloc[idx]
        ct_cropped = get_ct_cropped_disk_cache(candidateInfo.name, candidateInfo['xyzCoord'])
        ct_cropped = torch.tensor(ct_cropped).unsqueeze(0) # add batch dimension
        isNodule_label = candidateInfo['isNodule']
        # one_hot_encoding_tensor = F.one_hot(labels).to(torch.float32)
        one_hot_encoding_tensor = torch.tensor([0, 1]) if isNodule_label else torch.tensor([1, 0])
        return ct_cropped, one_hot_encoding_tensor

# training and validation datasets classes, which share the same parent self.df_candidates, 
# but the shared df_candidates is splitted
class LunaDataset_Train(LunaDataset):
    def __init__(self):
        super().__init__()
        self.df_candidates = self.df_candidates[:self.frac_split_idx]
        
class LunaDataset_Val(LunaDataset):
    def __init__(self):
        super().__init__()
        self.df_candidates = self.df_candidates[self.frac_split_idx:]