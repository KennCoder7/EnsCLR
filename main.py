#!/usr/bin/env python
import argparse
import sys
import os
import shutil
import zipfile
import time
import torch
# torchlight
import torchlight
from torchlight import import_class

from processor.processor import init_seed
init_seed(0)

def save_src(target_path):
    code_root = os.getcwd()
    srczip = zipfile.ZipFile('./src.zip', 'w')
    for root, dirnames, filenames in os.walk(code_root):
            for filename in filenames:
                if filename.split('\n')[0].split('.')[-1] == 'py':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'yaml':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'ipynb':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
    srczip.close()
    save_path = os.path.join(target_path, 'src_%s.zip' % time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
    shutil.copy('./src.zip', save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    # processors['linear_evaluation'] = import_class('processor.linear_evaluation.LE_Processor')
    # processors['linear_evaluation_single'] = import_class('processor.linear_evaluation_single.LE_Processor')
    # processors['linear_evaluation_finetune'] = import_class('processor.linear_evaluation_finetune.LE_Processor')
    # processors['linear_evaluation_fusionclr'] = import_class('processor.le_fusionclr.LE_Processor')
    # processors['linear_evaluation_aimclr'] = import_class('processor.le_aimclr.LE_Processor')
    # processors['linear_evaluation_fusion_crossclr3s'] = import_class('processor.le_fusion_crossclr3s.LE_Processor')
    # processors['linear_evaluation_fusion_crossclr3s_origin'] = import_class('processor.le_fusion_crossclr3s_origin.LE_Processor')
    # processors['pretrain_crossclr_3views'] = import_class('processor.pretrain_crossclr_3views.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr'] = import_class('processor.pretrain_crossclr.CrosSCLR_Processor')
    # processors['pretrain_skeletonclr'] = import_class('processor.pretrain_skeletonclr.SkeletonCLR_Processor')
    # processors['train_stgcn'] = import_class('processor.train_stgcn.STGCN_Processor')
    # processors['pretrain_aimclr'] = import_class('processor.pretrain_aimclr.AimCLR_Processor')
    # processors['pretrain_partsclr'] = import_class('processor.pretrain_partsclr.Pretrain_Processor')
    # processors['pretrain_partsclr_reback'] = import_class('processor.pretrain_partsclr_reback.Pretrain_Processor')
    # processors['pretrain_partsclr_reback_1'] = import_class('processor.pretrain_partsclr_reback_1.Pretrain_Processor')
    # processors['pretrain_partsclr_reback_2'] = import_class('processor.pretrain_partsclr_reback_2.Pretrain_Processor')
    # processors['pretrain_partsclr_reback_3'] = import_class('processor.pretrain_partsclr_reback_3.Pretrain_Processor')
    # processors['pretrain_partsclr_slowfast'] = import_class('processor.pretrain_partsclr_slowfast.Pretrain_Processor')
    # processors['pretrain_partsclr_aimclr'] = import_class('processor.pretrain_partsclr_aimclr.Pretrain_Processor')
    # processors['pretrain_partsclr_aimclr_af'] = import_class('processor.pretrain_partsclr_aimclr_af.Pretrain_Processor')
    # processors['pretrain_partsclr_aimclr_af_0423'] = import_class('processor.pretrain_partsclr_aimclr_af_0423.Pretrain_Processor')
    # processors['pretrain_partsclr_aimclr_af_0425'] = import_class('processor.pretrain_partsclr_aimclr_af_0425.Pretrain_Processor')
    # processors['pretrain_partsclr_aimclr_af_0425_2'] = import_class('processor.pretrain_partsclr_aimclr_af_0425_2.Pretrain_Processor')
    # processors['pretrain_partsclr_aimclr_tl'] = import_class('processor.pretrain_partsclr_aimclr_tl.Pretrain_Processor')
    # processors['pretrain_partsclr_aimclr_tl_nnm'] = import_class('processor.pretrain_partsclr_aimclr_tl_nnm.Pretrain_Processor')
    # processors['pretrain_partsclr_aimclr_origin'] = import_class('processor.pretrain_partsclr_aimclr_origin.Pretrain_Processor')
    # processors['pretrain_partsclr_aimclr_olaf'] = import_class('processor.pretrain_partsclr_aimclr_olaf.Pretrain_Processor')
    #
    # processors['pretrain_partsclr_skeletonclr'] = import_class('processor.pretrain_partsclr_skeletonclr.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_bf'] = import_class('processor.pretrain_partsclr_skeletonclr_bf.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_f'] = import_class('processor.pretrain_partsclr_skeletonclr_f.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_ftl'] = import_class('processor.pretrain_partsclr_skeletonclr_ftl.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_context'] = import_class('processor.pretrain_partsclr_skeletonclr_context.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_context_tl'] = import_class('processor.pretrain_partsclr_skeletonclr_context_tl.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_origin'] = import_class('processor.pretrain_partsclr_skeletonclr_origin.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_distill'] = import_class('processor.pretrain_partsclr_skeletonclr_distill.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_distill_bf'] = import_class('processor.pretrain_partsclr_skeletonclr_distill_bf.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_olaf'] = import_class('processor.pretrain_partsclr_skeletonclr_olaf.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_olaf_mm'] = import_class('processor.pretrain_partsclr_skeletonclr_olaf_mm.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_olaf_sc'] = import_class('processor.pretrain_partsclr_skeletonclr_olaf_sc.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_skeletonclr_olaf_ss'] = import_class('processor.pretrain_partsclr_skeletonclr_olaf_ss.SkeletonCLR_Processor')
    #
    # # endregion yapf: enable
    # processors['pretrain_partsclr_aim_origin'] = import_class('processor.pretrain_partsclr_aim_origin.Pretrain_Processor')
    # processors['pretrain_partsclr_aim_af'] = import_class('processor.pretrain_partsclr_aim_af.Pretrain_Processor')
    # processors['pretrain_partsclr_aim_olaf'] = import_class('processor.pretrain_partsclr_aim_olaf.Pretrain_Processor')
    #
    # processors['pretrain_partsclr_crossclr_olaf'] = import_class('processor.pretrain_partsclr_crossclr_olaf.SkeletonCLR_Processor')
    # processors['pretrain_partsclr_crossclr3s_olaf'] = import_class('processor.pretrain_partsclr_crossclr3s_olaf.SkeletonCLR_Processor')
    #
    # processors['pretrain_vicreg_skeletonclr'] = import_class('processor.pretrain_vicreg_skeletonclr.SkeletonCLR_Processor')

    # processors['train_fusion_stgcn'] = import_class('processor.train_fusion_stgcn.STGCN_Processor')
    # processors['train_fusion_stgcn_1'] = import_class('processor.train_fusion_stgcn_1.STGCN_Processor')
    # processors['train_fusion_stgcn_1_1'] = import_class('processor.train_fusion_stgcn_1_1.STGCN_Processor')
    # processors['train_fusion_stgcn_2'] = import_class('processor.train_fusion_stgcn_2.STGCN_Processor')
    # processors['pretrain_fusion_crossclr_3views'] = import_class('processor.pretrain_fusion_crossclr_3views.Pretrain_Processor')
    # processors['pretrain_fusion_crossclr_3views_re'] = import_class('processor.pretrain_fusion_crossclr_3views_re.Pretrain_Processor')
    # processors['train_stack_fusion_stgcn'] = import_class('processor.train_stack_fusion_stgcn.STGCN_Processor')

    # processors['pretrain_crossclr_3views'] = import_class('processor.pretrain_crossclr_3views.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_2'] = import_class('processor.pretrain_crossclr_3views_2.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_4'] = import_class('processor.pretrain_crossclr_3views_4.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_5'] = import_class('processor.pretrain_crossclr_3views_5.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_5_a'] = import_class('processor.pretrain_crossclr_3views_5_a.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_aimclr'] = import_class('processor.pretrain_crossclr_3views_aimclr.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_4_b'] = import_class('processor.pretrain_crossclr_3views_4_b.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_4_i'] = import_class('processor.pretrain_crossclr_3views_4_i.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_fusion'] = import_class('processor.pretrain_crossclr_3views_fusion.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_fusion_re'] = import_class('processor.pretrain_crossclr_3views_fusion_re.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_fusion_v_1'] = import_class('processor.pretrain_crossclr_3views_fusion_v_1.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_fusion_v_2'] = import_class('processor.pretrain_crossclr_3views_fusion_v_2.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_fusion_v_0'] = import_class('processor.pretrain_crossclr_3views_fusion_v_0.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_fusion_v_3'] = import_class('processor.pretrain_crossclr_3views_fusion_v_3.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_fusion_v_4'] = import_class('processor.pretrain_crossclr_3views_fusion_v_4.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_fusion_v_4_a'] = import_class('processor.pretrain_crossclr_3views_fusion_v_4_a.CrosSCLR_3views_Processor')
    # processors['pretrain_crossclr_3views_fusion_v_3_c'] = import_class('processor.pretrain_crossclr_3views_fusion_v_3_c.CrosSCLR_3views_Processor')

    # processors['pretrain_ensemble_clr'] = import_class('processor.pretrain_ensemble_clr.Pretrain_Processor')
    # processors['le_ensemble_clr'] = import_class('processor.le_ensembleclr.LE_Processor')
    # processors['pretrain_ensemble_aimclr'] = import_class('processor.pretrain_ensemble_aimclr.Pretrain_Processor')
    # processors['pretrain_ensemble_skeletonclr'] = import_class('processor.pretrain_ensemble_skeletonclr.Pretrain_Processor')
    # processors['pretrain_ensemble_aimclr_pro'] = import_class('processor.pretrain_ensemble_aimclr_pro.Pretrain_Processor')
    # processors['pretrain_ensemble_aimclr_single'] = import_class('processor.pretrain_ensemble_aimclr_single.Pretrain_Processor')
    # processors['pretrain_ensemble_aimclr_single_p2'] = import_class('processor.pretrain_ensemble_aimclr_single_p2.Pretrain_Processor')
    # processors['pretrain_ensemble_aimclr_multiple'] = import_class('processor.pretrain_ensemble_aimclr_multiple.Pretrain_Processor')
    # processors['pretrain_ensemble_aimclr_v'] = import_class('processor.pretrain_ensemble_aimclr_v.Pretrain_Processor')
    # processors['pretrain_ensemble_aimclr_v_online'] = import_class('processor.pretrain_ensemble_aimclr_v_online.Pretrain_Processor')
    # processors['pretrain_ensemble_aimclr_v_online_1'] = import_class('processor.pretrain_ensemble_aimclr_v_online_1.Pretrain_Processor')
    # processors['pretrain_ensemble_aimclr_v_online_2'] = import_class('processor.pretrain_ensemble_aimclr_v_online_2.Pretrain_Processor')
    # processors['visualize_ensemble_aimclr_single'] = import_class('processor.visualize_ensemble_aimclr_single.Pretrain_Processor')
    # processors['visualize_skeletonclr'] = import_class('processor.visualize_skeletonclr.Pretrain_Processor')

    processors['pretrain_ensemble_aimclr'] = import_class('processor.pretrain_ensemble_aimclr.Pretrain_Processor')
    processors['linear_evaluation'] = import_class('processor.linear_evaluation.LE_Processor')



    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    arg.device = torch.cuda.current_device()
    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    if p.arg.phase == 'train':
        # save src
        save_src(p.arg.work_dir)

    p.start()
