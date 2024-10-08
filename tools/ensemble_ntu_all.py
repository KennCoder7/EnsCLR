import pickle
import numpy as np
from tqdm import tqdm

# Linear
print('-' * 20 + 'Linear Eval' + '-' * 20)

# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro/joint_xsub_le_joint_0603/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro/bone_xsub_le_bone_0603/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro/motion_xsub_le_motion_0603/'

# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro/joint_xsub_le_all_0603/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro/bone_xsub_le_all_0603/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro/motion_xsub_le_all_0603/'
# ms_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr/xsub_le_all_0603/'
# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr/xsub_le_joint_0603/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr/xsub_le_bone_0603/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr/xsub_le_motion_0603/'

# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/joint_xsub_le_joint1/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/bone_xsub_le_bone1/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/motion_xsub_le_motion1/'

# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/joint_xsub_le_all/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/bone_xsub_le_all/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/motion_xsub_le_all/'

# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/joint_xview_le_joint1/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/bone_xview_le_bone1/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/motion_xview_le_motion1/'

# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/joint_xview_le_all/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/bone_xview_le_all/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus/motion_xview_le_all/'
# ms_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr/xview_le_all/'


# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_ntu120/xsub_le_joint/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_ntu120/xsub_le_bone/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_ntu120/xsub_le_motion/'

# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/joint_xsub_le_joint1/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/bone_xsub_le_bone1/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/motion_xsub_le_motion1/'

joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/joint_xsub_le_all/'
bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/bone_xsub_le_all/'
motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/motion_xsub_le_all/'
ms_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_ntu120/xsub_le_all/'

# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/joint_xsetup_le_joint1/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/bone_xsetup_le_bone1/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/motion_xsetup_le_motion1/'

# joint_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/joint_xsetup_le_all/'
# bone_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/bone_xsetup_le_all/'
# motion_path = '/nfs/users/wangkun/slowfast/SklCLR/PARTSCLR/output/ensemble_aimclr/ensemble_aimclr_single_pro_plus_ntu120/motion_xsetup_le_all/'

# label = open('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/xsub/val_label.pkl', 'rb')
# label = open('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D/xview/val_label.pkl', 'rb')
label = open('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D-120/xsub/val_label.pkl', 'rb')
# label = open('/nfs/users/wangkun/datasets/NTU/NTU-RGB-D-120/xsetup/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(joint_path + 'test_result.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open(bone_path + 'test_result.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open(motion_path + 'test_result.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open(ms_path + 'test_result.pkl', 'rb')
r4 = list(pickle.load(r4).items())


weights = [0.4, 0.4, 0.4, 0.6]  # joint, bone, motion, ms
# weights = [0.5, 0.5, 0.5]  # joint, bone, motion

right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    _, r44 = r4[i]

    # print(r11.shape, r22.shape, r33.shape, r44.shape, r55.shape, r66.shape, r77.shape)
    r = r11 * weights[0] + r22 * weights[1] + r33 * weights[2] + r44 * weights[3]
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print('top1: ', acc)
print('top5: ', acc5)
