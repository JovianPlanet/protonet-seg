'''
Function to register NFBS MRI dataset to the MRBrains13 coordinate space
'''

import os
from pathlib import Path
from ants import image_read, image_write, registration, apply_transforms

def ants_registration(reg_type, 
        ref_mri=None,  # Reference MRI
        reg_mri='', 
        unreg_dir=None,  # Unregistered MRIs folder 
        out_dir=None): # Output folder

    if ref_mri == None:
        ref_mri = os.path.join('/media', 'davidjm', 'Disco_Compartido', 'david', 'datasets', 'MRBrainS13DataNii', 'TrainingData', '2', 'T1.nii')

    if unreg_dir == None:
        unreg_dir = os.path.join('/media', 'davidjm', 'Disco_Compartido', 'david', 'datasets', 'NFBS_Dataset')

    if out_dir == None:
        out_dir = os.path.join('/media', 'davidjm', 'Disco_Compartido', 'david', 'datasets', 'NFBS_Registered')

    if reg_mri:
        reg_mri = '_'+reg_mri
    else:
        reg_mri = ''

    '''
    Se itera sobre las subcarpetas del directorio
    '''

    mri_dir = next(os.walk(unreg_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: working directory

    ref = image_read(ref_mri) # Load reference image #, pixeltype='unsigned char')

    for directory in mri_dir[:]:

        if reg_type == 'mask':
            filename = 'sub-' + directory + '_ses-NFB3_T1w_brainmask.nii.gz'
            img_NFBS = image_read(os.path.join(unreg_dir, directory, filename))#, pixeltype='unsigned char')
            rs2_reg = registration(fixed=ref, moving=img_NFBS, type_of_transform = 'DenseRigid' )
            rs2 = apply_transforms(fixed=ref, moving=img_NFBS, transformlist=rs2_reg['fwdtransforms'], interpolator='multiLabel')

        elif reg_type == 'brain':
            filename = 'sub-'+directory+'_ses-NFB3_T1w_brain.nii.gz'
            img_NFBS = image_read(os.path.join(unreg_dir, directory, filename))#, pixeltype='unsigned char')
            rs2_reg = registration(fixed=ref, moving=img_NFBS, type_of_transform = 'DenseRigid' )
            rs2 = apply_transforms(fixed=ref, moving=img_NFBS, transformlist=rs2_reg['fwdtransforms'])

        elif reg_type == 'head':
            filename = 'sub-'+directory+'_ses-NFB3_T1w.nii.gz'
            img_NFBS = image_read(os.path.join(unreg_dir, directory, filename))#, pixeltype='unsigned char')
            rs2_reg = registration(fixed=ref, moving=img_NFBS, type_of_transform = 'DenseRigid' )
            rs2 = apply_transforms(fixed=ref, moving=img_NFBS, transformlist=rs2_reg['fwdtransforms'])
            
        Path(os.path.join(out_dir, directory)).mkdir(parents=True, exist_ok=True)
        image_write(rs2, os.path.join(out_dir, directory, 'REG_' + filename), ri=False)
        print(f'Volume {directory} registered and saved to disk\n')

    print('All volumes registered!\n')