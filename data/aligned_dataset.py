#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase)

        # ------------------------------
        ####### Total-Text Dataset ######
        # ------------------------------
        # image_path = 'Dataset/totaltext/Images'
        # char_mask_path = 'Groundtruth/Pixel/Character Level Mask/groundtruth_pixel'
        # txt_bb_mask_path = 'Groundtruth/Pixel/Text Region Mask/groundtruth_textregion/Text_Region_Mask'
        # Phase : Train / Test

        self.A_paths = sorted(make_dataset(self.dir_A))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        # -------------------------------
        self.has_mask_GT = False
        # if 'total-text' in opt.name:
        if 'text' in opt.mask_type: # Incase train on SVT and test on TotalText mask
            self.has_mask_GT = True
            self.tottxt_image_path = 'Dataset/totaltext/Images'
            self.tottxt_char_mask_path = 'Groundtruth/Pixel/Character Level Mask/groundtruth_pixel'
            self.tottxt_txt_bb_mask_path = 'Groundtruth/Pixel/Text Region Mask/groundtruth_textregion/Text_Region_Mask'

            transform_list = [transforms.ToTensor()]
            self.transform_mask = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        w, h = A.size
        if self.has_mask_GT:
            tmp = A_path.split(self.tottxt_image_path)
            A_char_mask_path = ''.join([tmp[0], self.tottxt_char_mask_path,tmp[1]])
            # print(A_char_mask_path)
            A_txt_bb_mask_path = ''.join([tmp[0], self.tottxt_txt_bb_mask_path,tmp[1]])[:-3] + 'png' # jpg to png

            A_char_mask = Image.open(A_char_mask_path)
            A_txt_bb_mask = Image.open(A_txt_bb_mask_path)

        if w < h:
            ht_1 = self.opt.loadSize * h // w
            wd_1 = self.opt.loadSize
        else:
            wd_1 = self.opt.loadSize * w // h
            ht_1 = self.opt.loadSize
        A = A.resize((wd_1, ht_1), Image.BICUBIC)
        A = self.transform(A)

        if self.has_mask_GT:
            A_char_mask = A_char_mask.resize((wd_1, ht_1), Image.BICUBIC)
            A_txt_bb_mask = A_txt_bb_mask.resize((wd_1, ht_1), Image.BICUBIC)
            A_char_mask = self.transform_mask(A_char_mask)
            A_txt_bb_mask = self.transform_mask(A_txt_bb_mask)

        h = A.size(1)
        w = A.size(2)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        # -----------tottxt ---------------
        if self.has_mask_GT:
            if 'random' in self.opt.mask_type: # randomly shifted mask from original text place.
                w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
                h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
            A_char_mask = A_char_mask[:, h_offset:h_offset + self.opt.fineSize,
                   w_offset:w_offset + self.opt.fineSize]
            A_txt_bb_mask = A_txt_bb_mask[:, h_offset:h_offset + self.opt.fineSize,
                   w_offset:w_offset + self.opt.fineSize]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)] # size(2)-1, size(2)-2, ... , 0
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)

        # let B directly equals A
        B = A.clone()
        out =  {'A': A, 'B': B,
                'A_paths': A_path}
        if self.has_mask_GT:
            out['A_char_mask'] = A_char_mask
            out['A_txt_bb_mask'] = A_txt_bb_mask
        return out

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
