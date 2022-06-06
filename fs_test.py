import torch
from torch.utils.data import DataLoader
from datasets.dataloader import FewShot_Dataloader
from datasets.fewshot_sampler import NShotTaskSampler
from models.unet import UnetEncoder
from models.metrics import * 
from utils.plots import *


PATH_SUPERVISED = './models/fs_weights.pth'
TEST_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/test'

n_test = 5 # n shots (test)
k_test = 1 # k way (k classes) (test)
q_test = 5 # q queries (test)

episodes_per_epoch = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

test_ds = FewShot_Dataloader(
    TEST_PATH,
    'T1.nii', 
    'LabelsForTraining.nii',
    48, 
    'testing'
)

test_mris = DataLoader(
    test_ds, 
    batch_sampler=NShotTaskSampler(test_ds, episodes_per_epoch, n_test, k_test, q_test),
)

supp = n_test*k_test
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

        idx_b = y_s==0 # Pertenecen al background
        idx_1 = y_s==1 # Pertenecen a la clase
        idx_b = idx_b.unsqueeze(axis=1)
        idx_1 = idx_1.unsqueeze(axis=1)

        f_s_mask = f_s*idx_1

        proto = torch.sum(f_s_mask, dim=(0, 2, 3))
        sum_ = torch.sum(idx_1)
        if sum_ == 0:
            proto = proto*0.0
        else:
            proto = proto / sum_
        proto = proto.view(1, proto.shape[0], 1, 1)

        dists = cosine_dist(proto, f_q)

        pval = torch.where(dists>0.8, 1, 0)
        dice = dice_coeff(pval, y_q)
        running_dice += dice
        #print(f'{dice.item()=:.3f}')
        plot_batch_full(x_q.squeeze(1), y_q, pval>0.8)

print(f'Val dice = {running_dice/(j + 1):.3f}\n')