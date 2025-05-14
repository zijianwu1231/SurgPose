import os
import re
import cv2
import pickle
import random
import torch

import numpy as np

from pathlib import Path
from typing import List, Tuple, Dict
from .augmentor_mix_dataset import Augmentor as Augmentor
from .pinhole_augmentor_mix_dataset import Augmentor as PinholeAugmentor
from torch.utils.data import Dataset
from lz4.frame import decompress as lzdecompress
from PIL import Image
from core.utils import frame_utils

class MixDataset(Dataset):
    def __init__(self,
                 mode:str,
                 camera_type:str,
                 dataset_name: List[str],
                 data_path: Dict,
                 fields: List[List[str]],
                 filelists: List[str],
                 input_size: Tuple[int, int],
                 ):
        super().__init__()
        self.data = []
        self.camera_type = camera_type
        self.dataset_name = dataset_name
        self.data_paths = data_path
        self.height = input_size[0]
        self.width = input_size[1]

        if mode == 'train':

            if camera_type == 'fisheye':
                self.augmentor = Augmentor(
                    image_height=self.height,
                    image_width=self.width,
                    max_disp=self.width,
                    scale_min=0.6,
                    scale_max=1.0,
                    seed=0,
                    camera_type=self.camera_type,
                    albumentations_aug=False,
                    white_balance_aug=False,
                    rgb_noise_aug=False,
                    motion_blur_aug=False,
                    local_blur_aug=False,
                    global_blur_aug=False,
                    chromatic_aug=True,
                    camera_motion_aug=True
                )

            else:
                self.augmentor = PinholeAugmentor(
                    image_height=self.height,
                    image_width=self.width,
                    max_disp=self.width,
                    scale_min=0.6,
                    scale_max=1.0,
                    seed=0,
                    camera_type=self.camera_type,
                    albumentations_aug=False,
                    white_balance_aug=False,
                    rgb_noise_aug=False,
                    motion_blur_aug=False,
                    local_blur_aug=False,
                    global_blur_aug=False,
                    chromatic_aug=True,
                    camera_motion_aug=False
                )
        else:
            self.augmentor = None
            # if camera_type == 'fisheye':
            #     self.augmentor = Augmentor(
            #         image_height=self.height,
            #         image_width=self.width,
            #         max_disp=self.width,
            #         scale_min=0.6,
            #         scale_max=1.0,
            #         seed=0,
            #         camera_type=self.camera_type,
            #         albumentations_aug=True,
            #         white_balance_aug=False,
            #         rgb_noise_aug=False,
            #         motion_blur_aug=False,
            #         local_blur_aug=False,
            #         global_blur_aug=False,
            #         chromatic_aug=True,
            #         camera_motion_aug=False
            #     )
            # else:
            #     self.augmentor = PinholeAugmentor(
            #         image_height=self.height,
            #         image_width=self.width,
            #         max_disp=self.width,
            #         scale_min=0.6,
            #         scale_max=1.0,
            #         seed=0,
            #         camera_type=self.camera_type,
            #         albumentations_aug=False,
            #         white_balance_aug=False,
            #         rgb_noise_aug=False,
            #         motion_blur_aug=False,
            #         local_blur_aug=False,
            #         global_blur_aug=False,
            #         chromatic_aug=False,
            #         camera_motion_aug=False
            #     )

        self.rng = np.random.RandomState(0)

        for i in range(len(data_path['image'])):

            name = self.dataset_name[i]
            field = fields[i]
            filelist = filelists[i]

            if 'realistic' in name or 'far' in name or 'samba' in name or 'cre' in name \
                or name in ['fallingthings','HR_VS','InStereo2K','sceneflow','sintel','TartanAir','vkitti2']:
                items = [line.strip().split(',') for line in open(filelist).readlines()[1:]]
            else:
                items = [line.strip().split(' ') for line in open(filelist).readlines()[:]]

            for item in items:
                assert len(item) >= 3, f'dataset: {name}, item: {item}'
            if 'far' in name:

                items = [{
                    'name': name, 'left': item[field.index('left')],
                    'right': item[field.index('right')],
                } for item in items]
            else:
                # print(name, field, items[0])

                items = [{
                    'name': name, 'left': item[field.index('left')],
                    'right': item[field.index('right')],
                    'left_disp': item[field.index('left_disp')],
                    'right_disp': None if 'right_disp' not in field else item[field.index('right_disp')],
                    'seg_sky': None if 'seg_sky' not in field else item[field.index('seg_sky')],
                    'seg_wire': None if 'seg_wire' not in field else item[field.index('seg_wire')]
                } for item in items]

            weight = 1
            # print(weight, name, len(weight * [(i, item) for item in items]))

            self.data = self.data + weight * [(i, item) for item in items]

    def read_pfm(self, file):
        """ Read a pfm file """
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        header = str(bytes.decode(header, encoding='utf-8'))
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        pattern = r'^(\d+)\s(\d+)\s$'
        temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
        dim_match = re.match(pattern, temp_str)
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            temp_str += str(bytes.decode(file.readline(), encoding='utf-8'))
            dim_match = re.match(pattern, temp_str)
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception('Malformed PFM header: width, height cannot be found')

        scale = float(file.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        # DEY: I don't know why this was there.
        # sometimes need flip
        data = np.flipud(data)
        file.close()
        return data, scale

    def load_pkl(self, pkl_path):
        data = pklload(pkl_path)
        return_data = {}
        for k, v in data.items():
            if k not in ["shape", "dtype"]:
                v = decompress(v, data["dtype"])
                v = v.reshape(data["shape"])
            return_data[k] = v
        return return_data

    def get_disp(self, path, input_h, input_w, dataset_name):
        error = None
        if 'airsim' in dataset_name:
            # disp = np.load(path)['data'].astype(np.float32)
            # keys = np.load(path).files
            # disp = np.load(path)[keys[0]].astype(np.float32)

            if 'fog' in dataset_name:
                disp = np.ones((input_h, input_w)) * 1e-6

            else:

                disps = self.load_pkl(path)
                disp = disps["data"].astype(np.float32)

        elif 'realistic' in dataset_name:
            # print(np.load(path).files)
            # keys = np.load(path).files
            # disp = np.load(path)[keys[0]].astype(np.float32)
            # error = np.load(path)[keys[1]].astype(np.float32)

            if 'far' in dataset_name:
                disp = np.ones((input_h, input_w)) * 1e-6
            else:
                disps = self.load_pkl(path)

                if '3d_recon' in dataset_name or 'lidar' in dataset_name:
                    # disp = disps["data"].astype(np.float32)
                    disp = disps["disp"].astype(np.float32) # pseudo
                elif 'depth' in dataset_name:
                    disp = disps['data'].astype(np.float32)

                else:
                    # 一致性检验
                    disp = disps["disp"].astype(np.float32)
                    error = disps["error"].astype(np.float32) if "error" in disps else None

            '''
            # 全局伪标签
            disp = disps["disp"].astype(np.float32)
            error = disps["error"].astype(np.float32)
            '''

        # pinhole
        elif 'stereo_trainset' in dataset_name:
            disp = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 32
        elif 'UnrealStereo' in dataset_name:
            disp, scale = self.read_pfm(path)

            
        elif 'cre' in dataset_name:
            disp = frame_utils.readDispCREStereo(path)
        elif 'InStereo2K' in dataset_name:
            disp, _ = frame_utils.readDispInStereo2K(path)
        elif 'TartanAir' in dataset_name:
            disp, _ = frame_utils.readDispTartanAir(path)
        elif 'fallingthings' in dataset_name:
            disp, _ = frame_utils.readDispFallingThings(path)
        elif 'sceneflow' in dataset_name:
            disp = frame_utils.read_gen(path)
        elif 'sintel' in dataset_name:
            disp, _ = frame_utils.readDispSintelStereo(path)
        elif 'vkitti2' in dataset_name:
            disp, _ = frame_utils.readDispVKITTI2(path)
        elif 'HR_VS' in dataset_name:
            disp = frame_utils.read_gen(path)        
        else:
            print("Dataset can't be recognize!")
        return disp, error

    def __getitem__(self, index):
        # find path
        while True:
            try:
                path_index, item = self.data[index]
                dataset_name = item['name']
                image_root_path = self.data_paths['image'][path_index]
                disp_root_path = self.data_paths['disp'][path_index]
                seg_root_path = self.data_paths['seg'][path_index]

                seg_sky = None
                seg_wire = None
                if 'realistic' in dataset_name:
                    left = os.path.join(image_root_path, item['left'])
                    if '_wire_' in dataset_name:
                        right = os.path.join(disp_root_path, item['right'])
                    else:
                        right = os.path.join(image_root_path, item['right'])

                    if 'far' not in dataset_name:
                        left_disp = os.path.join(disp_root_path, item['left_disp'])
                        right_disp = None if item['right_disp'] == None else os.path.join(disp_root_path,
                                                                                          item['right_disp'])
                        seg_sky = None if item['seg_sky'] == None else os.path.join(seg_root_path,
                                                                                    item['seg_sky'])
                        seg_wire = None if item['seg_wire'] == None else os.path.join(seg_root_path,
                                                                                    item['seg_wire'])

                else:
                    left = os.path.join(image_root_path, item['left'])
                    right = os.path.join(image_root_path, item['right'])

                    if 'far' not in dataset_name:
                        left_disp = os.path.join(disp_root_path, item['left_disp'])
                        right_disp = None if item['right_disp'] == None else os.path.join(disp_root_path, item['right_disp'])
                        seg_sky = None if item['seg_sky'] == None else os.path.join(seg_root_path, item['seg_sky'])
                        seg_wire = None if item['seg_wire'] == None else os.path.join(seg_root_path, item['seg_wire'])

                # read img, disp
                assert os.path.exists(left), f'dataset: {dataset_name}, root: {image_root_path}, left_path: {left}'
                left_img = cv2.imread(left, cv2.IMREAD_COLOR)
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                right_img = cv2.imread(right, cv2.IMREAD_COLOR)
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
                # left_img = np.array(Image.open(left).convert('RGB'))
                # right_img = np.array(Image.open(right).convert('RGB'))

                if 'far' not in dataset_name:
                    left_disp, left_error = self.get_disp(left_disp, left_img.shape[0], left_img.shape[1], dataset_name)

                    right_disp, right_error = right_disp, None \
                        if right_disp == None else self.get_disp(right_disp, left_img.shape[0], left_img.shape[1],
                                                                 dataset_name)

                else:
                    left_disp, left_error = self.get_disp('', left_img.shape[0], left_img.shape[1], dataset_name)
                    right_disp, right_error = '', None

                if dataset_name in ['TartanAir', 'vkitti2', 'sintel']:
                    left_img = cv2.resize(left_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                    right_img = cv2.resize(right_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                    left_disp = cv2.resize(left_disp, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR) * 1.5
                # must before flip
                wire_mask = np.ones_like(left_disp)
                if seg_wire and os.path.exists(seg_wire):

                    if 'samba' in dataset_name:

                        gt_wire_key = self.load_pkl(seg_wire)
                        gt_wire = gt_wire_key["data"].astype(np.float32)

                        wire_mask[gt_wire == 2] = 10
                    else:
                        gt_wire_key = np.load(seg_wire).files[0]
                        gt_wire = np.load(seg_wire)[gt_wire_key]
                        # print(gt_sky.sum()/(gt_sky.shape[0]*gt_sky.shape[1]))
                        wire_mask[gt_wire==1] = 10

                # airsim samba seg
                if 'samba' in dataset_name and 'noseg' not in dataset_name:
                    gt_seg = self.load_pkl(seg_sky)["data"].astype(np.float32)

                    wire_mask[gt_seg == 2] = 10

                # if seg_wire and not os.path.exists(seg_wire):
                #     print('Wire seg path error: {}'.format(seg_wire))

                if self.augmentor is not None and self.rng.binomial(1, 0.5) == 1 and isinstance(right_disp, np.ndarray):
                    left_img, right_img = np.fliplr(right_img), np.fliplr(left_img)
                    left_disp, right_disp = np.fliplr(right_disp), np.fliplr(left_disp)
                    wire_mask = np.fliplr(wire_mask)

                left_disp[left_disp == np.inf] = 0.0
                left_disp[left_disp == -1] = 0.0  # real data无效数据为-1
                left_disp[np.isnan(left_disp)] = 0.

                if self.rng.binomial(1, 0.5) == 1 and isinstance(right_disp, np.ndarray) and self.camera_type != 'pinhole':
                    left_img, right_img = np.flipud(left_img), np.flipud(right_img)
                    left_disp, right_disp = np.flipud(left_disp), np.flipud(right_disp)
                    wire_mask = np.flipud(wire_mask)
                
                if self.augmentor is not None:
                    left_img, right_img, right_img_ori, left_disp, disp_mask, wire_mask = self.augmentor(
                        dataset_name, left_img, right_img, left_disp, left_error, wire_mask
                    ) 
                else:
                    disp_mask = (left_disp < 512) & (left_disp > 0)


                # if seg_sky and os.path.exists(seg_sky) and 'group1' in left:
                #     from .transform import calculate_iou
                #     iou = calculate_iou(gt_sky[::2, ...], disp_mask)
                #     print(iou, left)

                # left_img = left_img.transpose(2, 0, 1).astype("uint8")
                # right_img = right_img.transpose(2, 0, 1).astype("uint8")
                # right_img_ori = right_img_ori.transpose(2, 0, 1).astype("uint8")
                left_img = torch.from_numpy(left_img).permute(2, 0, 1).float()
                right_img = torch.from_numpy(right_img).permute(2, 0, 1).float()
                left_disp = torch.from_numpy(left_disp).unsqueeze(0).float()
                disp_mask = torch.from_numpy(disp_mask).float()
                
                if self.augmentor is not None and (left_img.shape != (3, self.height, self.width) or right_img.shape != (3, self.height, self.width)):
                    index = random.randint(0, self.__len__())
                    print('shape error: ', left_img.shape, right_img.shape, left)
                    continue

                # return {
                #     "left": left_img,
                #     "right": right_img,
                #     "right_ori": right_img_ori,
                #     "disparity": left_disp,
                #     "mask": disp_mask,
                #     'left_path': left,
                #     'wire_mask': wire_mask,
                # }
                
                # return left_img, right_img, left_disp, left
                return left, left_img, right_img, left_disp, disp_mask
            except Exception as e:
                print('Excep when:', index, e)
                print(left)
                index = random.randint(0, self.__len__())
                continue

    def __len__(self):
        return len(self.data)

def pklload(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def decompress(data, dtype):
    data = lzdecompress(data)
    data = np.frombuffer(data, dtype=dtype)
    return data
