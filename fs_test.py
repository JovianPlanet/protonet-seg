import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from datasets.dataloader import FewShot_Dataloader
from datasets.fewshot_sampler import NShotTaskSampler
from models.unet import UnetEncoder
from models.metrics import * 
from utils.plots import *

#PATH_SUPERVISED = './models/best/fsmulti_weights-jun24.pth'
PATH_SUPERVISED = './models/fsmul_wts-cross-h8-ep25-10_.pth'

TEST_PATH = '/media/davidjm/Disco_Compartido/david/datasets/MRBrainS-All/test'

n_test = 5 # n shots (test)
k_test = 1 # k way (k classes) (test)
q_test = 3 # q queries (test)

episodes_per_epoch = 1

classes = ['GM', 'WM', 'CSF']
num_classes = len(classes)

num_heads = 2

classes_dic = {'BGR': 0, 'GM': 1, 'WM': 2, 'CSF': 3}
batch_dice = {'GM': None, 'WM': None, 'CSF': None}
gen_dice = {'GM': 0.0, 'WM': 0.0, 'CSF': 0.0}

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
unet = UnetEncoder(1, depth=5, start_filts=32).to(device, dtype=torch.double)
unet.load_state_dict(torch.load(PATH_SUPERVISED))
#print(f'{summary(unet.double(), (18, 1, 240, 240))}')

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

        pv = get_class_prototypes(f_s, y_s)

        dv = get_dists(f_q, pv)
        mv = torch.argmax(dv, dim=1)

        for c in classes: #range(mv.shape[0]): # in enumerate(dv):

            batch_dice[c] = dice_coeff(torch.where(mv==classes_dic[c], 1, 0), 
                              torch.where(y_q==classes_dic[c], 1, 0)
            )

            gen_dice[c] += batch_dice[c].item()

            plot_batch_full(x_q, y_q, mv)

gen_dice = {k: v / (j+1) for k, v in gen_dice.items()}
print(f'{gen_dice.values()}, {sum(gen_dice.values())/num_classes:.3f}')
