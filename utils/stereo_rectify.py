# This script is borrowed from the following repository:
# Repository Name: robust-pose-estimator
# Repository Link: https://github.com/aimi-lab/robust-pose-estimator
import cv2
import numpy as np
from typing import Tuple
import os
import configparser
import json
import torch
import warnings

def get_rect_maps(
    lcam_mat = None, 
    rcam_mat = None, 
    rmat = None,
    tvec = None,
    ldist_coeffs = None,
    rdist_coeffs = None,
    img_size: Tuple[int, int] = (1280, 1024),
    triangular_intrinsics: bool = False,
    mode: str = 'conventional'
    ) -> dict:
    if mode == 'conventional':
        if triangular_intrinsics:
            lcam_mat = np.array([[lcam_mat[0, 0], 0, lcam_mat[0, 2]], [0, lcam_mat[1, 1], lcam_mat[1, 2]], [0, 0, 1]], dtype=np.float64)
            rcam_mat = np.array([[rcam_mat[0, 0], 0, rcam_mat[0, 2]], [0, rcam_mat[1, 1], rcam_mat[1, 2]], [0, 0, 1]], dtype=np.float64)

        # compute pixel mappings
        r1, r2, p1, p2, q, valid_pix_roi1, valid_pix_roi2 = cv2.stereoRectify(cameraMatrix1=lcam_mat.astype('float64'), distCoeffs1=ldist_coeffs.astype('float64'),
                                                                            cameraMatrix2=rcam_mat.astype('float64'), distCoeffs2=rdist_coeffs.astype('float64'),
                                                                            imageSize=tuple(img_size), R=rmat.astype('float64'), T=tvec.T.astype('float64'),
                                                                            alpha=0)

        lmap1, lmap2 = cv2.initUndistortRectifyMap(cameraMatrix=lcam_mat, distCoeffs=ldist_coeffs, R=r1, newCameraMatrix=p1, size=tuple(img_size), m1type=cv2.CV_32FC1)
        rmap1, rmap2 = cv2.initUndistortRectifyMap(cameraMatrix=rcam_mat, distCoeffs=ldist_coeffs, R=r2, newCameraMatrix=p2, size=tuple(img_size), m1type=cv2.CV_32FC1)
        maps = {'lmap1': lmap1,
                'lmap2': lmap2,
                'rmap1': rmap1,
                'rmap2': rmap2}
    elif mode == 'pseudo':
        maps = {}
        p1 = lcam_mat.astype('float64')
        p2 = rcam_mat.astype('float64')
    else:
        raise NotImplementedError

    return maps, p1, p2


def rectify_pair(limg, rimg, maps, method='nearest'):

    cv_interpol = cv2.INTER_NEAREST if method == 'nearest' else cv2.INTER_CUBIC

    limg_rect = cv2.remap(np.copy(limg), maps['lmap1'], maps['lmap2'], interpolation=cv_interpol)
    rimg_rect = cv2.remap(np.copy(rimg), maps['rmap1'], maps['rmap2'], interpolation=cv_interpol)

    return limg_rect, rimg_rect

def pseudo_rectify(rimg, x0, x1):

    tmat = np.array(((1, 0, x0-x1), (0, 1, 0))).astype(np.float32)
    rimg_rect = cv2.warpAffine(rimg, tmat, (rimg.shape[1], rimg.shape[0]))

    return rimg_rect

def pseudo_rectify_2d(rimg, x0, x1, y0, y1):

    tmat = np.array(((1, 0, x0-x1), (0, 1, y0-y1))).astype(np.float32)
    rimg_rect = cv2.warpAffine(rimg, tmat, (rimg.shape[1], rimg.shape[0]))

    return rimg_rect

class StereoRectifier(object):
    def __init__(self, calib_file, img_size_new=None, mode='conventional'):
        if os.path.splitext(calib_file)[1] == '.json':
            cal = self._load_calib_json(calib_file)
        elif os.path.splitext(calib_file)[1] == '.ini':
            cal = self._load_calib_ini(calib_file)
        elif os.path.splitext(calib_file)[1] == '.yaml':
            cal = self._load_calib_yaml(calib_file)
        else:
            raise NotImplementedError

        assert mode in ['conventional', 'pseudo']
        self.mode = mode
        if self.mode =='pseudo':
            warnings.warn('pseudo rectification used', UserWarning)

        self.scale = 1.0
        if img_size_new is not None:
            # scale intrinsics
            self.scale = img_size_new[0]/cal['img_size'][0]
            print('scale factor:', self.scale)
            h_crop = int((cal['img_size'][1]*self.scale - img_size_new[1])/2)
            assert h_crop >= 0, 'only vertical crop implemented'
            cal['lkmat'][:2] *= self.scale
            cal['rkmat'][:2] *= self.scale
            cal['lkmat'][1, 2] -= h_crop
            cal['rkmat'][1, 2] -= h_crop
            cal['img_size'] = img_size_new
        self.img_size = cal['img_size']
        self.cal = cal

        self.maps, self.l_intr, self.r_intr = get_rect_maps(
            lcam_mat=cal['lkmat'],
            rcam_mat=cal['rkmat'],
            rmat=cal['R'],
            tvec=cal['T'],
            ldist_coeffs=cal['ld'],
            rdist_coeffs=cal['rd'],
            img_size=tuple(map(round, cal['img_size'])), #cal['img_size'],
            mode=self.mode
        )

    def __call__(self, img_left, img_right):
        img_left = img_left.permute(1,2,0).numpy()
        img_right = img_right.permute(1, 2, 0).numpy()
        if self.mode == 'pseudo':
            x0, x1, y0, y1 = self.cal['lkmat'][0][-1], self.cal['rkmat'][0][-1], self.cal['lkmat'][1][-1], self.cal['rkmat'][1][-1]
            img_right_rect = pseudo_rectify_2d(img_right, x0, x1, y0, y1)
            img_left_rect = img_left
        else:
            img_left_rect, img_right_rect = rectify_pair(img_left, img_right, self.maps)
        img_left_rect = torch.tensor(img_left_rect).permute(2,0,1)
        img_right_rect = torch.tensor(img_right_rect).permute(2,0,1)
        return img_left_rect, img_right_rect

    def get_rectified_calib(self):
        calib_rectifed = {'intrinsics': {}}
        calib_rectifed['intrinsics']['left'] = self.l_intr[:3,:3]
        calib_rectifed['intrinsics']['right'] = self.r_intr[:3,:3]
        calib_rectifed['extrinsics'] = np.eye(4)
        if self.mode == 'conventional':
            calib_rectifed['extrinsics'][:3,3] = np.array([self.r_intr[0, 3] / self.r_intr[0, 0], 0., 0.]) # Tx*f, see cv2 website
        else:
            calib_rectifed['extrinsics'][:3,3] = self.cal['T']
        calib_rectifed['bf'] = np.sqrt(np.sum(calib_rectifed['extrinsics'][:3, 3] ** 2))*self.l_intr[0, 0] # baseline * focal_length
        calib_rectifed['bf_orig'] = calib_rectifed['bf']/self.scale
        calib_rectifed['img_size'] = self.img_size
        return calib_rectifed

    @staticmethod
    def _load_calib_json(fname):

        with open(fname, 'rb') as f: json_dict = json.load(f)

        lkmat = np.eye(3)
        lkmat[0, 0] = json_dict['data']['intrinsics'][0]['f'][0]
        lkmat[1, 1] = json_dict['data']['intrinsics'][0]['f'][1]
        lkmat[:2, -1] = json_dict['data']['intrinsics'][0]['c']

        rkmat = np.eye(3)
        rkmat[0, 0] = json_dict['data']['intrinsics'][1]['f'][0]
        rkmat[1, 1] = json_dict['data']['intrinsics'][1]['f'][1]
        rkmat[:2, -1] = json_dict['data']['intrinsics'][1]['c']

        ld = np.array(json_dict['data']['intrinsics'][0]['k'])
        rd = np.array(json_dict['data']['intrinsics'][1]['k'])

        tvec = np.array(json_dict['data']['extrinsics']['T'])
        rmat = cv2.Rodrigues(np.array(json_dict['data']['extrinsics']['om']))[0]

        img_size = (json_dict['data']['width'], json_dict['data']['height'])

        cal = {}
        cal['lkmat'] = lkmat
        cal['rkmat'] = rkmat
        cal['ld'] = ld
        cal['rd'] = rd
        cal['T'] = tvec
        cal['R'] = rmat
        cal['img_size'] = img_size

        return cal

    @staticmethod
    def _load_calib_ini(fname):
        config = configparser.ConfigParser()
        config.read(fname)
        img_size = (float(config['StereoLeft']['res_x']), float(config['StereoLeft']['res_y']))

        lkmat = np.eye(3)
        lkmat[0, 0] = float(config['StereoLeft']['fc_x'])
        lkmat[1, 1] = float(config['StereoLeft']['fc_y'])
        lkmat[0, 2] = float(config['StereoLeft']['cc_x'])
        lkmat[1, 2] = float(config['StereoLeft']['cc_y'])

        rkmat = np.eye(3)
        rkmat[0, 0] = float(config['StereoRight']['fc_x'])
        rkmat[1, 1] = float(config['StereoRight']['fc_y'])
        rkmat[0, 2] = float(config['StereoRight']['cc_x'])
        rkmat[1, 2] = float(config['StereoRight']['cc_y'])

        ld = np.array([float(config['StereoLeft']['kc_0']),float(config['StereoLeft']['kc_1']),
                       float(config['StereoLeft']['kc_2']),float(config['StereoLeft']['kc_3']),float(config['StereoLeft']['kc_4']),
                       float(config['StereoLeft']['kc_5']), float(config['StereoLeft']['kc_6']),float(config['StereoLeft']['kc_7'])])
        rd = np.array([float(config['StereoRight']['kc_0']),float(config['StereoRight']['kc_1']),
                       float(config['StereoRight']['kc_2']),float(config['StereoRight']['kc_3']),float(config['StereoRight']['kc_4']),
                       float(config['StereoRight']['kc_5']), float(config['StereoRight']['kc_6']),float(config['StereoRight']['kc_7'])])

        tvec = np.array([float(config['StereoRight']['T_0']),float(config['StereoRight']['T_1']),float(config['StereoRight']['T_2'])])
        rmat = np.zeros((3,3))
        rmat[0, 0] = float(config['StereoRight']['R_0'])
        rmat[0, 1] = float(config['StereoRight']['R_1'])
        rmat[0, 2] = float(config['StereoRight']['R_2'])
        rmat[1, 0] = float(config['StereoRight']['R_3'])
        rmat[1, 1] = float(config['StereoRight']['R_4'])
        rmat[1, 2] = float(config['StereoRight']['R_5'])
        rmat[2, 0] = float(config['StereoRight']['R_6'])
        rmat[2, 1] = float(config['StereoRight']['R_7'])
        rmat[2, 2] = float(config['StereoRight']['R_8'])

        cal = {}
        cal['lkmat'] = lkmat
        cal['rkmat'] = rkmat
        cal['ld'] = ld
        cal['rd'] = rd
        cal['T'] = tvec
        cal['R'] = rmat
        cal['img_size'] = img_size

        return cal

    @staticmethod
    def _load_calib_yaml(fname):
        fs = cv2.FileStorage(fname, cv2.FILE_STORAGE_READ)
        img_size = (int(fs.getNode('Camera.width').real()), int(fs.getNode('Camera.height').real()))

        lkmat = fs.getNode('M1').mat()
        rkmat = fs.getNode('M2').mat()

        ld = fs.getNode('D1').mat()
        rd = fs.getNode('D2').mat()

        tvec = fs.getNode('T').mat()
        rmat = fs.getNode('R').mat()

        cal = {}
        cal['lkmat'] = lkmat
        cal['rkmat'] = rkmat
        cal['ld'] = ld
        cal['rd'] = rd
        cal['T'] = tvec
        cal['R'] = rmat
        cal['img_size'] = img_size

        return cal
