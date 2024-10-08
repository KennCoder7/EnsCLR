import numpy as np
import os
from tqdm import tqdm
train_label = np.load('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/train_label.npy')
train_position = np.load('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/train_position.npy')
val_label = np.load('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/val_label.npy')
val_position = np.load('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/val_position.npy')


if not os.path.exists('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/train_sampler'):
    os.makedirs('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/train_sampler')
if not os.path.exists('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/val_sampler'):
    os.makedirs('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/val_sampler')

for i in tqdm(range(len(train_position))):
    np.save('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/train_sampler/{}.npy'.format(i), train_position[i])
for i in tqdm(range(len(val_position))):
    np.save('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/NTU60_frame50/xsub/val_sampler/{}.npy'.format(i), val_position[i])
