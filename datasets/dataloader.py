import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib

class FewShot_Dataloader(Dataset):
    def __init__(self, path, mri_name, label_name, slices, phase): #(self, config, phase):

        assert phase in ['training', 'validating', 'testing']
        self.phase = phase
        self.path = path
        self.label_dict = { 'GM' : 1, 'BG' : 2, 'WM' : 3, 'WML' : 4,
                            'CSF' : 5, 'VEN' : 6, 'CER' : 7, 'BSTEM' : 8}

        self.subjects = next(os.walk(self.path))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?

        self.L = []

        if phase == 'training':

            for i, subject in enumerate(self.subjects[:4]):
                mri_path = os.path.join(self.path, subject, mri_name)
                label_path = os.path.join(self.path, subject, label_name)
                label_ = np.int16(nib.load(label_path).get_data())
                label_ = np.where(label_ == self.label_dict['GM'], label_, 0)

                for slice_ in range(slices):
                    if np.sum(label_[:, :, slice_]) > 0:
                        self.L.append([subject, slice_, mri_path, label_path])

            self.df = pd.DataFrame(self.L, columns=['Subject', 'Slice', 'Path MRI', 'Path Label'])

        elif phase == 'validating':
            for i, subject in enumerate(self.subjects[4]):
                mri_path = os.path.join(self.path, subject, mri_name)
                label_path = os.path.join(self.path, subject, label_name)

                for slice_ in range(slices):
                    self.L.append([subject, slice_, mri_path, label_path])

            self.df = pd.DataFrame(self.L, columns=['Subject', 'Slice', 'Path MRI', 'Path Label'])


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if self.phase == 'training':

            load_path = self.df.at[index, 'Path MRI']
            mri = np.int16(nib.load(load_path).get_data())
            load_slice = self.df.at[index, 'Slice']
            label_ = np.int16(nib.load(self.df.at[index, 'Path Label']).get_data())
            label = np.where(label_ == self.label_dict['GM'], label_, 0)
            #label = label + np.where(label_ == self.label_dict['WM'], label_, 0)
            #label = label + np.where(label_ == self.label_dict['CSF'], label_, 0)
            return mri[:, :, load_slice], label[:, :, load_slice]#, load_path, load_slice

        elif self.phase == 'validating':
            load_path = self.df.at[index, 'Path MRI']
            mri = nib.load(load_path).get_data()
            load_slice = self.df.at[index, 'Slice']
            label_ = np.int16(nib.load(self.df.at[index, 'Path Label']).get_data())
            label = np.where(label_ == self.label_dict['GM'], label_, 0)
            #label = label + np.where(label_ == self.label_dict['WM'], label_, 0)
            #label = label + np.where(label_ == self.label_dict['CSF'], label_, 0)
            return mri[:, :, load_slice], label[:, :, load_slice]#, load_path, load_slice

        elif self.phase == 'testing':
            pass #return self.patches[index], self.whole_vol


class unlabeled_Dataloader(Dataset):
    def __init__(self, path, slices, phase): #(self, config, phase):

        assert phase in ['training', 'validating', 'testing']
        self.phase = phase
        self.path = path
        self.name_template = 'REG_sub-_ses-NFB3_T1w.nii.gz' #sub-A00056306_ses-NFB3_T1w_brain # REG_sub-A00033747_ses-NFB3_T1w

        if phase == 'training':

            subjects = next(os.walk(self.path))[1] # [2]: lists files; [1]: lists subdirectories; [0]: ?

            L = []

            for i, subject in enumerate(subjects):
                mri_path = os.path.join(self.path, subject, self.name_template.replace('_ses', subject + '_ses'))


                for slice_ in range(slices):
                    L.append([subject, slice_, mri_path])

            self.df = pd.DataFrame(L, columns=['Subject', 'Slice', 'Path MRI'])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if self.phase == 'training':

            load_path = self.df.at[index, 'Path MRI']
            mri = nib.load(load_path).get_data()#.transpose(0,2,1)
            load_slice = self.df.at[index, 'Slice']
            return mri[:, :, load_slice], load_path, load_slice

        if self.phase == 'validating':
            pass #return self.patches[index], self.label[index], self.whole_vol

        if self.phase == 'testing':
            pass #return self.patches[index], self.whole_vol
