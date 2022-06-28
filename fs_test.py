import torch
from torch.utils.data import DataLoader
from datasets.dataloader import FewShot_Dataloader
from datasets.fewshot_sampler import NShotTaskSampler
from models.unet import UnetEncoder
from models.metrics import * 
from utils.plots import *

PATH_SUPERVISED = './models/fsmulti_weights.pth'
TEST_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/test'

n_test = 5 # n shots (test)
k_test = 1 # k way (k classes) (test)
q_test = 5 # q queries (test)

episodes_per_epoch = 1

classes = ['GM', 'WM', 'CSF']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

torch.cuda.empty_cache()

test_ds = FewShot_Dataloader(
    TEST_PATH,
    'T1.nii', 
    'LabelsForTraining.nii',
    48, 
    'testing'
)

test_mris = DataLoader(
    test_ds, 
    batch_sampler=NShotTaskSampler(test_ds, 
                                   episodes_per_epoch, 
                                   n_test, 
                                   k_test, 
                                   q_test,
                                   fixed_tasks=[classes]
                                  ),
)

supp = len(classes)*n_test # n_test*k_test
unet = UnetEncoder(1, depth=5).to(device, dtype=torch.double)
unet.load_state_dict(torch.load(PATH_SUPERVISED))

# since we're not training, we don't need to calculate the gradients for our outputs
unet.eval()
with torch.no_grad():
    running_dice = 0.0
    for j, data in enumerate(test_mris):
                
        x, y = data
        x = x.unsqueeze(1).to(device, dtype=torch.double)
        y = y.to(device, dtype=torch.double)

        y_s = y[:supp].to(device)

        x_q = x[supp:]
        y_q = y[supp:].to(device)

        f = unet(x)

        f_s = f[:supp]
        f_q = f[supp:]

        d1 = get_prototype_all(f_s, y_s, f_q)

        dice = 0.0
        for k, d in enumerate(d1[1:]):

            dice += dice_coeff(d[k*n_test:(k*n_test)+n_test,:,:], 
                               y_q[k*n_test:(k*n_test)+n_test,:,:].double()
            )

            plot_batch_full(x_q.squeeze(1)[k*n_test:(k*n_test)+n_test,:,:], 
                            y_q[k*n_test:(k*n_test)+n_test,:,:], 
                            d[k*n_test:(k*n_test)+n_test,:,:]>0.5
            )

        running_dice += dice/len(classes)

print(f'Val dice = {running_dice/(j + 1):.3f}\n')