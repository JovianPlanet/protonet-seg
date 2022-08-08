import torch
from torch.utils.data import DataLoader
from datasets.dataloader import FewShot_Dataloader
from datasets.fewshot_sampler import NShotTaskSampler
from models.unet import UnetEncoder
from models.metrics import * 
from utils.plots import *

#PATH_SUPERVISED = './models/best/fsmulti_weights-jun24.pth'
PATH_SUPERVISED = './models/fsmul_wts-dice-h8-ep50-10.pth'

TEST_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/test'

n_test = 5 # n shots (test)
k_test = 1 # k way (k classes) (test)
q_test = 3 # q queries (test)

episodes_per_epoch = 1

classes = ['GM', 'WM', 'CSF']
num_heads = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

torch.cuda.empty_cache()

test_ds = FewShot_Dataloader(
    TEST_PATH,
    'T1.nii', 
    'LabelsForTraining.nii',
    48, 
    num_heads
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

        print(f'{y_q.shape=}')

        f = unet(x)

        f_s = f[:supp]
        f_q = f[supp:]

        d1 = get_prototype_all(f_s, y_s, f_q, n_test, q_test)

        dice = 0.0
        for k, d in enumerate(d1):

            x = x_q.squeeze(1)[k*q_test:(k*q_test)+q_test,:,:]
            y = torch.where(y_q[k*q_test:(k*q_test)+q_test,:,:]>0.0, 1.0, 0.0)

            dice += dice_coeff(d>0.9, y.double())

            plot_batch_full(x, y, d>0.9)

        running_dice += dice/len(classes)

print(f'Val dice = {running_dice/(j + 1):.3f}\n')