import cv2
import numpy as np
from PIL import Image, ImageEnhance
from .transform import *
from core.utils.geometry import eulerAnglesToRotationMatrix

class Augmentor:
    def __init__(
            self,
            image_height=384,
            image_width=512,
            max_disp=150,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
            camera_type='pinhole',
            albumentations_aug=True,
            white_balance_aug=True,
            rgb_noise_aug=True,
            motion_blur_aug=True,
            local_blur_aug=True,
            global_blur_aug=True,
            chromatic_aug=True,
            camera_motion_aug=True

    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)
        self.camera_type = camera_type
        self.carlib_auger = OpticShiftAugmentor(height=1408, width=1280, fov_h=185, fov_w=120, maxDegree=1)

        self.albumentations_aug = albumentations_aug
        self.white_balance_aug = white_balance_aug
        self.rgb_noise_aug = rgb_noise_aug
        self.motion_blur_aug = motion_blur_aug
        self.local_blur_aug = local_blur_aug
        self.global_blur_aug = global_blur_aug
        self.chromatic_aug = chromatic_aug
        self.camera_motion_aug = camera_motion_aug

    def chromatic_augmentation(self, img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

        img = Image.fromarray(img)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        gamma_map = [
                        255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
                    ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img_ = np.array(img)

        return img_

    def padding(self, data, pad_len):
        # top_ext = data[-pad_len:]
        bott_ext = data[0: pad_len]
        return np.concatenate((data, bott_ext), axis=0)

    def __call__(self, dataset_name, left_img, right_img, left_disp, error=None, wire_mask=None):

        # 2740 -> 1408
        crop_h = 1408
        if 'airsim' in dataset_name:
            rescale_h = left_img.shape[0]

            # crop
            # dx = self.rng.randint(direc - random_len, direc + random_len)
            dx = self.rng.randint(0, rescale_h)

            if (dx + crop_h) > rescale_h:
                pad_len = (dx + crop_h) - rescale_h
                left_img = self.padding(left_img, pad_len)
                right_img = self.padding(right_img, pad_len)
                left_disp = self.padding(left_disp, pad_len)
                wire_mask = self.padding(wire_mask, pad_len)
            # else:
            #     pass

            left_img = left_img[dx: dx + crop_h, ...]
            right_img = right_img[dx:dx + crop_h, ...]
            left_disp = left_disp[dx:dx + crop_h, ...]
            wire_mask = wire_mask[dx:dx + crop_h, ...]

        # 1. chromatic augmentation
        # left_img = self.chromatic_augmentation(left_img)
        # right_img = self.chromatic_augmentation(right_img)
        right_img_ori = right_img.copy()

        # disp mask
        resize_scale = 1.0
        # disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
        if error is not None:
            # error_mask = error/(left_disp+1e-6) <1.0
            error_mask = error < 0.5
            # error_mask_int = error_mask.astype(np.uint8)
            # print(error_mask_int.sum(), error_mask_int.sum()/(error_mask.shape[0]*error_mask.shape[1]))
            disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0) & error_mask
        else:
            disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
        # disp_mask = disp_mask.astype("float32")

        if self.local_blur_aug and self.rng.binomial(1, 0.2):

            # 左图模糊增广
            p_l = self.rng.binomial(1, 0.5)
            if p_l < 0.5:
                brightness = self.rng.uniform(low=-40, high=40)
                mask_l = self.rng.choice([None, 'local_mask'])
                if mask_l == 'local_mask':
                    mask_l = mask_ge(left_img.shape, self.rng, weights=[0.5, 0.5])
                left_img, _ = image_blur_mask(left_img, self.rng, mask_l, brightness)

            # 右图模糊增广
            p_r = self.rng.binomial(1, 0.5)
            if p_r < 0.5:
                mask_r = self.rng.choice([None, 'local_mask'])
                brightness = self.rng.uniform(low=-40, high=40)
                if mask_r == 'local_mask':
                    mask_r = mask_ge(left_img.shape, self.rng, weights=[0.5, 0.5])
                right_img, _ = image_blur_mask(right_img, self.rng, mask_r, brightness)

        if self.rgb_noise_aug and self.rng.binomial(1, 0.5):
            sigma = self.rng.uniform(low=1, high=5)
            left_img = RGB_noise_aug(left_img, sigma, self.rng)
            right_img = RGB_noise_aug(right_img, sigma, self.rng)

        if self.chromatic_aug and self.rng.binomial(1, 0.4):
            # left_img, right_img = chromatic_augmentation_v3(
            #     left_img.astype(np.uint8), right_img.astype(np.uint8),
            #     self.rng, self.motion_blur_aug, self.albumentations_aug
            # )
            left_img = self.chromatic_augmentation(left_img)
            right_img = self.chromatic_augmentation(right_img)

        # Diff chromatic # White balance
        if self.white_balance_aug and self.rng.binomial(1, 0.5):
            random_ratio_L = self.rng.uniform(-0.3, 0.3)
            random_ratio_R = self.rng.uniform(-0.15, 0.15) + random_ratio_L
            left_img = white_balance_augmentation(left_img, ratio=random_ratio_L)
            right_img = white_balance_augmentation(right_img, ratio=random_ratio_R)

                # global aug # 模拟失焦
        if self.global_blur_aug and self.rng.binomial(1, 0.2):

            # 左图模糊增广
            p_l = self.rng.binomial(1, 0.5)
            if p_l < 0.5:
                kernel_size = self.rng.randint(2, 7) * 2 + 1
                left_img, _ = image_blur_all(left_img, (kernel_size, kernel_size))

            # 右图模糊增广
            p_r = self.rng.binomial(1, 0.5)
            if p_r < 0.5:
                # kernel = self.rng.randint(5, 15)
                kernel_size = self.rng.randint(2, 7) * 2 + 1
                right_img, _ = image_blur_all(right_img, (kernel_size, kernel_size))

        # 2. spatial augmentation
        # 2.1) rotate & vertical shift for right image
        if self.camera_motion_aug and self.rng.binomial(1, 0.2):
            if 'airsim' in dataset_name:
                sigma = 0.2
            else:
                sigma = 0.08
            right_img = self.carlib_auger.shift(right_img, self.rng, sigma)
            
            # angle, pixel = 0.1, 2
            # px = self.rng.uniform(-pixel, pixel)
            # ag = self.rng.uniform(-angle, angle)
            # image_center = (
            #     self.rng.uniform(0, right_img.shape[0]),
            #     self.rng.uniform(0, right_img.shape[1]),
            # )
            # rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            # right_img = cv2.warpAffine(
            #     right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            # )
            # trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            # right_img = cv2.warpAffine(
            #     right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            # )


        # 1408 -> 704
        crop_h_2 = 704
        start = self.rng.randint(0, crop_h - crop_h_2)
        left_img = left_img[start: start+crop_h_2, ...]
        right_img = right_img[start: start+crop_h_2, ...]
        left_disp = left_disp[start: start+crop_h_2, ...]
        right_img_ori = right_img_ori[start: start+crop_h_2, ...]
        disp_mask = disp_mask[start: start+crop_h_2, ...]
        wire_mask = wire_mask[start: start+crop_h_2, ...]

        # random occlusion
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx: cx + sx, cy - sy: cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]

        # color mask 过滤掉过曝, 欠曝的像素
        # color_mask = np.logical_and(np.mean(left_img, axis=2) < 235, np.mean(left_img, axis=2) > 20)
        # disp_mask = np.logical_and(disp_mask, color_mask)

        return left_img, right_img, right_img_ori, left_disp, disp_mask, wire_mask

class OpticShiftAugmentor:
    def __init__(self, height, width, fov_h=360, fov_w=180, maxDegree=1):
        # print("Optic Shift Augmentor on Cassini projection")
        self.maxRad = np.deg2rad(maxDegree)
        self.height = height
        self.width = width

        self.fov_h = np.deg2rad(fov_h)
        self.fov_w = np.deg2rad(fov_w)
        self.fov_h_2 = self.fov_h / 2
        self.fov_w_2 = self.fov_w / 2

        theta_1_start = self.fov_h_2 - (self.fov_h_2 / self.height)
        theta_1_end = -self.fov_h_2
        theta_1_step = 2 * self.fov_h_2 / self.height
        self.theta_1_step = theta_1_step
        theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
        theta_1_map = np.array([theta_1_range for i in range(self.width)]).astype(np.float32).T

        phi_1_start = self.fov_w_2 - (self.fov_w_2 / self.width)
        phi_1_end = -self.fov_w_2
        phi_1_step = 2 * self.fov_w_2 / self.width
        self.phi_1_step = phi_1_step
        phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
        phi_1_map = np.array([phi_1_range for j in range(self.height)]).astype(np.float32)

        x = np.sin(phi_1_map)
        y = np.cos(phi_1_map) * np.sin(theta_1_map)
        z = np.cos(phi_1_map) * np.cos(theta_1_map)
        self.X_2 = np.expand_dims(np.dstack((x, y, z)), axis=-1)

    def shift(self, rightImg, rng, sigma):
        mu, sigma = 0, sigma  # mean and standard deviation
        # angle_deg = np.fmod(rng.normal(mu, sigma, size=3), 1) #
        angle_deg_x = np.fmod(rng.normal(mu, sigma, size=1), 3)/10
        # angle_deg_y = np.fmod(rng.normal(mu, sigma, size=1), 1)/10
        angle_deg_y = np.zeros_like(angle_deg_x)
        angle_deg_z = np.fmod(rng.normal(mu, sigma, size=1), 1)/10
        angle_deg = np.array([angle_deg_x, angle_deg_y, angle_deg_z])

        max_vshift = 2
        vshift = np.random.uniform(-max_vshift, max_vshift)
        # ag_0, ag_1, ag_2 = 2,2,2 # debug
        # vshift = 0 # debug
        # print([ag_0, ag_1, ag_2, vshift]) # debug
        angle_rad = np.deg2rad(angle_deg)

        RotMatrix = eulerAnglesToRotationMatrix(angle_rad)

        # apply rotation and compute LUT
        X_rot = np.matmul(RotMatrix, self.X_2)
        r_2 = np.sqrt(np.square(X_rot[:, :, 0, 0]) + np.square(X_rot[:, :, 1, 0]) + np.square(X_rot[:, :, 2, 0]))
        theta_2_map = np.arctan2(X_rot[:, :, 1, 0], X_rot[:, :, 2, 0]).astype(np.float32)
        phi_2_map = np.arcsin(np.clip(X_rot[:, :, 0, 0] / r_2, -1, 1)).astype(np.float32)
        LUT_x = -phi_2_map * self.width / self.fov_w - 0.5 + self.width / 2
        LUT_y = -theta_2_map * self.height / self.fov_h - 0.5 + self.height / 2 + vshift

        # remap
        right_shift = cv2.remap(rightImg, LUT_x, LUT_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        return right_shift

class Augmentor_v0():
    def __init__(
            self,
            image_height=384,
            image_width=512,
            max_disp=150,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
            camera_type='pinhole',
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)
        self.camera_type = camera_type

    def chromatic_augmentation(self, img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

        img = Image.fromarray(img)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        gamma_map = [
                        255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
                    ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img_ = np.array(img)

        return img_

    def padding(self, data, pad_len):
        top_ext = data[-pad_len:]
        bott_ext = data[0:pad_len]
        return np.concatenate((top_ext, data, bott_ext), axis=0)

    def __call__(self, dataset_name, left_img, right_img, left_disp, error=None):
        # 1. chromatic augmentation
        left_img = self.chromatic_augmentation(left_img)
        right_img_ori = right_img.copy()
        right_img = self.chromatic_augmentation(right_img)

        # 2. spatial augmentation
        # 2.1) rotate & vertical shift for right image
        if self.rng.binomial(1, 0.5):
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (
                self.rng.uniform(0, right_img.shape[0]),
                self.rng.uniform(0, right_img.shape[1]),
            )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(
                right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(
                right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )

        # 2.2) random resize
        resize_scale = 1.0
        # disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
        if error is not None:
            # error_mask = error/(left_disp+1e-6) <1.0
            error_mask = error < 0.5
            # error_mask_int = error_mask.astype(np.uint8)
            # print(error_mask_int.sum(), error_mask_int.sum()/(error_mask.shape[0]*error_mask.shape[1]))
            disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0) & error_mask
        else:
            disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
        disp_mask = disp_mask.astype("float32")

        # 2.3) random crop
        if 'airsim' in dataset_name:
            rescale_h = left_img.shape[0]

            # crop
            crop_h = 704
            # dx = self.rng.randint(direc - random_len, direc + random_len)
            dx = self.rng.randint(0, rescale_h)

            if dx + crop_h >= rescale_h:
                pad_len = rescale_h - (dx + crop_h)
                left_img = self.padding(left_img, pad_len)
                right_img = self.padding(right_img, pad_len)
                right_img_ori = self.padding(right_img_ori, pad_len)
                left_disp = self.padding(left_disp, pad_len)
                disp_mask = self.padding(disp_mask, pad_len)

            left_img = left_img[dx: dx + crop_h]
            right_img = right_img[dx:dx + crop_h]
            right_img_ori = right_img_ori[dx:dx + crop_h]
            left_disp = left_disp[dx:dx + crop_h]
            disp_mask = disp_mask[dx:dx + crop_h]

        # 3. add random occlusion to right image
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx: cx + sx, cy - sy: cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]

        return left_img, right_img, right_img_ori, left_disp, disp_mask