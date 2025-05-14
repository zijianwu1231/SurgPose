import cv2
import random
import numpy as np
import albumentations as A

from warnings import warn
from PIL import Image, ImageEnhance


def calculate_iou(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()

    iou = intersection / union
    return iou

def chromatic_augmentation(img):
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


def random_brightness_contrast(image, alpha, beta, beta_by_max=True):
    MAX_VALUES_BY_DTYPE = {
        np.dtype("uint8"): 255,
        np.dtype("uint16"): 65535,
        np.dtype("uint32"): 4294967295,
        np.dtype("float32"): 1.0,
    }

    if image.dtype == np.uint8:

        dtype = np.dtype("uint8")
        max_value = MAX_VALUES_BY_DTYPE[dtype]

        lut = np.arange(0, max_value + 1).astype("float32")

        if alpha != 1:
            lut *= alpha
        if beta != 0:
            if beta_by_max:
                lut += beta * max_value
            else:
                lut += (alpha * beta) * np.mean(image)

        lut = np.clip(lut, 0, max_value).astype(dtype)
        image = cv2.LUT(image, lut)
    else:
        dtype = image.dtype
        image = image.astype("float32")

        if alpha != 1:
            image *= alpha
        if beta != 0:
            if beta_by_max:
                max_value = MAX_VALUES_BY_DTYPE[dtype]
                image += beta * max_value
            else:
                image += beta * np.mean(image)
    return image


def random_gamma(image, gamma):
    if image.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        image = cv2.LUT(image, table.astype(np.uint8))
    else:
        image = np.power(image, gamma)

    return image


def random_hue_saturation(image, hue_shift, sat_shift, val_shift):
    def is_grayscale_image(image: np.ndarray) -> bool:
        return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)

    def clip(img: np.ndarray, dtype: np.dtype, maxval: float) -> np.ndarray:
        return np.clip(img, 0, maxval).astype(dtype)

    def _shift_hsv_uint8(img, hue_shift, sat_shift, val_shift):
        dtype = img.dtype
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(img)

        if hue_shift != 0:
            lut_hue = np.arange(0, 256, dtype=np.int16)
            lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)
            hue = cv2.LUT(hue, lut_hue)

        if sat_shift != 0:
            lut_sat = np.arange(0, 256, dtype=np.int16)
            lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype(dtype)
            sat = cv2.LUT(sat, lut_sat)

        if val_shift != 0:
            lut_val = np.arange(0, 256, dtype=np.int16)
            lut_val = np.clip(lut_val + val_shift, 0, 255).astype(dtype)
            val = cv2.LUT(val, lut_val)

        img = cv2.merge((hue, sat, val)).astype(dtype)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def _shift_hsv_non_uint8(img, hue_shift, sat_shift, val_shift):
        dtype = img.dtype
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue, sat, val = cv2.split(img)

        if hue_shift != 0:
            hue = cv2.add(hue, hue_shift)
            hue = np.mod(hue, 360)  # OpenCV fails with negative values

        if sat_shift != 0:
            sat = clip(cv2.add(sat, sat_shift), dtype, 1.0)

        if val_shift != 0:
            val = clip(cv2.add(val, val_shift), dtype, 1.0)

        img = cv2.merge((hue, sat, val))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    if hue_shift == 0 and sat_shift == 0 and val_shift == 0:
        return image

    is_gray = is_grayscale_image(image)
    if is_gray:
        if hue_shift != 0 or sat_shift != 0:
            hue_shift = 0
            sat_shift = 0
            warn(
                "HueSaturationValue: hue_shift and sat_shift are not applicable to grayscale image. "
                "Set them to 0 or use RGB image"
            )
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.dtype == np.uint8:
        image = _shift_hsv_uint8(image, hue_shift, sat_shift, val_shift)
    else:
        image = _shift_hsv_non_uint8(image, hue_shift, sat_shift, val_shift)

    if is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image


def motion_blur_v2(image, kernel):
    from typing import Callable
    from typing_extensions import Concatenate, ParamSpec
    P = ParamSpec("P")

    def get_num_channels(image: np.ndarray) -> int:
        return image.shape[2] if len(image.shape) == 3 else 1

    def _maybe_process_in_chunks(
            process_fn: Callable[Concatenate[np.ndarray, P], np.ndarray], **kwargs
    ) -> Callable[[np.ndarray], np.ndarray]:
        def __process_fn(img: np.ndarray) -> np.ndarray:
            num_channels = get_num_channels(img)
            if num_channels > 4:
                chunks = []
                for index in range(0, num_channels, 4):
                    if num_channels - index == 2:
                        # Many OpenCV functions cannot work with 2-channel images
                        for i in range(2):
                            chunk = img[:, :, index + i: index + i + 1]
                            chunk = process_fn(chunk, **kwargs)
                            chunk = np.expand_dims(chunk, -1)
                            chunks.append(chunk)
                    else:
                        chunk = img[:, :, index: index + 4]
                        chunk = process_fn(chunk, **kwargs)
                        chunks.append(chunk)
                img = np.dstack(chunks)
            else:
                img = process_fn(img, **kwargs)
            return img

        return __process_fn

    conv_fn = _maybe_process_in_chunks(cv2.filter2D, ddepth=-1, kernel=kernel)
    return conv_fn(image)


def get_motion_blur_kernel(blur_limit, allow_shifted):
    ksize = random.choice(np.arange(blur_limit[0], blur_limit[1] + 1, 2))
    if ksize <= 2:
        raise ValueError("ksize must be > 2. Got: {}".format(ksize))
    kernel = np.zeros((ksize, ksize), dtype=np.uint8)
    x1, x2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
    if x1 == x2:
        y1, y2 = random.sample(range(ksize), 2)
    else:
        y1, y2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)

    def make_odd_val(v1, v2):
        len_v = abs(v1 - v2) + 1
        if len_v % 2 != 1:
            if v2 > v1:
                v2 -= 1
            else:
                v1 -= 1
        return v1, v2

    if not allow_shifted:
        x1, x2 = make_odd_val(x1, x2)
        y1, y2 = make_odd_val(y1, y2)

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        center = ksize / 2 - 0.5
        dx = xc - center
        dy = yc - center
        x1, x2 = [int(i - dx) for i in [x1, x2]]
        y1, y2 = [int(i - dy) for i in [y1, y2]]

    cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
    # Normalize kernel
    return kernel.astype(np.float32) / np.sum(kernel)


def chromatic_augmentation_v2(img):
    brightness = A.Compose(
        [
            A.MotionBlur(blur_limit=7, allow_shifted=False, always_apply=False, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(40, 120), p=0.5),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=10, val_shift_limit=15)  # 色调

        ],
        p=1
    )
    return brightness(image=img)["image"]


def chromatic_augmentation_v3(l_img, r_img, rng, motion_blur_aug, albumentations_aug):
    # motion blur
    if motion_blur_aug and rng.binomial(1, 0.5):
        kernel = get_motion_blur_kernel((3, 7), allow_shifted=False)

        l_img = motion_blur_v2(l_img, kernel)
        r_img = motion_blur_v2(r_img, kernel)

    # random brightness contrast
    # if albumentations_aug and rng.binomial(1, 0.5):
    #     alpha = 1 + rng.uniform(-0.4, 0.4) # contrast
    #     beta = rng.uniform(-0.4, 0.4) # brightness
    #
    #     noise_alpha = rng.uniform(0.8, 1.2)
    #     noise_beta = rng.uniform(0.8, 1.2)
    #
    #     l_img = random_brightness_contrast(l_img, alpha, beta)
    #     r_img = random_brightness_contrast(r_img, alpha*noise_alpha, beta*noise_beta)

    # random gamma
    if albumentations_aug and rng.binomial(1, 0.5):
        gamma = rng.uniform(40, 300) / 100

        noise_gamma = rng.uniform(0.8, 1.2)

        l_img = random_gamma(l_img, gamma)
        r_img = random_gamma(r_img, gamma * noise_gamma)

    # hue saturation
    if albumentations_aug and rng.binomial(1, 0.5):
        hue_shift = rng.uniform(-20, 20)
        sat_shift = rng.uniform(-30, 30)
        val_shift = rng.uniform(-3, 3)

        noise_hue = rng.uniform(0.8, 1.2)
        noise_sat = rng.uniform(0.8, 1.2)
        noise_val = rng.uniform(0.8, 1.2)

        l_img = random_hue_saturation(l_img, hue_shift, sat_shift, val_shift)
        r_img = random_hue_saturation(r_img, hue_shift * noise_hue, sat_shift * noise_sat,
                                           val_shift * noise_val)

    return l_img, r_img


def white_balance_augmentation(img, ratio):
    assert img.shape[2] == 3
    # random_ratio = self.rng.uniform(-ratio, ratio)
    random_ratio = ratio
    img = np.asarray(img, dtype=np.float32)
    img[:, :, 0] *= np.asarray(1 + random_ratio, dtype=np.float32)
    img[:, :, 0] = img[:, :, 0].clip(min=0, max=255)  # .astype(np.uint8)
    return img


def RGB_noise_aug(image, sigma, rng):
    w, h = image.shape[1], image.shape[0]
    # print(image.shape)
    mu, sigma = 0, sigma
    gauss_noise = rng.normal(mu, sigma, (h, w, 3))
    image = image + gauss_noise
    image = np.clip(image, a_min=0, a_max=255)

    return image


def add_haze_v1(image, color, t=0.6, A=1):
    '''
        添加雾霾
        t : 透视率 0~1
        A : 大气光照
    '''
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr[0, 0] = color  # bgr
    out = image * t + A * bgr * (1 - t)
    return out


def disp2depth(disp):
    bf = 0.065 * 459
    return bf / (disp.clip(min=1e-3))


def random_haze_aug(image, disp, color=[210, 235, 255], A=0.95, x_beat=10):
    '''
           指定雾化中心
           A    : 大气光照  0.65-0.95
           beta : 雾化程度  0.01-0.08
           x_beat : 雾化程度  1-10
           colar: 雾霾颜色，[210-255, 235-255, 255] # b, g, r
       '''
    depth = disp2depth(disp)

    d = depth.clip(max=120)
    d = cv2.GaussianBlur(d, (5, 5), 0)

    # t = np.exp(-beta * d)
    # t = np.repeat(t[:, :, np.newaxis], 3, axis=2)

    beta = 0.00299846301309506 * np.exp(0.312484384678664 * x_beat)
    # print(beta)
    t = np.exp(-beta * d)
    t = np.repeat(t[:, :, np.newaxis], 3, axis=2)

    return add_haze_v1(image, color, t, A)


def motion_blur(image, degree, angle):
    '''
        生成运动模糊效果
        defree: 运动模糊程度,越大模糊程度越高
        angle: 运动模糊方向
    '''
    image = np.array(image)

    # 这里生成angle角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def low_illumination(image, gamma):
    '''
        生成弱光环境效果
        gamma:0.4-0.7,越小越暗
    '''
    lut = np.array([((i / 255.0) ** (1 / gamma)) * 255 for i in range(256)]).astype("uint8")

    # apply gamma correction using lookup table
    return cv2.LUT(image.astype(np.uint8), lut)


def image_blur_mask(img, rng, mask=None, brightness=40):
    '''
        在mask区域加入高斯模糊
        brightness: -40-40
    '''
    # 图像整体模糊
    scale_factor = 0.02  # 放缩倍数 0.02 - 0.1
    blurred = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    dsize = (img.shape[1], img.shape[0])
    blurred = cv2.resize(blurred, dsize, interpolation=cv2.INTER_CUBIC)
    blurred = blurred.astype(np.float32)

    if mask is None:
        w = rng.uniform(0.4, 0.6)
        blurred = img * (1 - w) + blurred * w
        return blurred.astype(np.uint8), blurred.astype(np.uint8)

    blurred = np.clip(blurred + brightness, 0, 255)

    result = img.astype(np.float32)
    mask = mask[:, :, np.newaxis]
    result = result * (1 - mask) + blurred * mask
    return result.astype(np.uint8), blurred.astype(np.uint8)


def image_blur_all(img, kernel_size=(15, 15)):
    '''
        全局高斯模糊
        kernel_size：5,5 - 15,15
    '''
    blur = cv2.GaussianBlur(img, kernel_size, 0)
    return blur, blur


def mask_ge(shape, rng, weights=[0.5, 0.5]):
    mask_model_list = ['random', 'ellipse']
    mask_model = rng.choice(mask_model_list, p=weights)

    # define mask size
    width = 30
    height = int(width * shape[0] / shape[1])

    if mask_model == 'random':
        # create random noise image
        noise = rng.randint(low=0, high=255, size=(height, width), dtype=np.uint8)

        # blur the noise image to control the size
        random_borderType = rng.choice([cv2.BORDER_CONSTANT, cv2.BORDER_CONSTANT])
        blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=3, sigmaY=3,
                                borderType=random_borderType)  # cv2.BORDER_CONSTANT

        stretch = blur

        # threshold stretched image to control the size
        thresh = cv2.threshold(stretch, 130, 255, cv2.THRESH_BINARY)[1]
        open_result = thresh

        # 开操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open_result = cv2.morphologyEx(open_result, cv2.MORPH_OPEN, kernel)

        # 高斯模糊
        result = cv2.GaussianBlur(open_result, (5, 5), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)
    else:
        width = 100
        height = int(width * shape[0] / shape[1])
        ellipse_height = rng.randint(low=int(width / 4),
                                     high=int(width / 2)) * 2 + 1  # Choose an odd kernel height between 3 and 29
        ellipse_width = rng.randint(low=int(width / 4),
                                    high=int(width / 2)) * 2 + 1  # Choose an odd kernel width between 3 and 29
        blur_pos = rng.randint(low=0, high=height), rng.randint(low=0, high=width)  # Choose a random position
        mask = np.zeros((height, width), dtype=np.uint8)
        # cv2.ellipse(mask, (blur_pos[1] + ellipse_width // 2, blur_pos[0] + ellipse_height // 2), (ellipse_width // 2, ellipse_height // 2), 0, 0, 360, 255, -1)
        angle = rng.randint(low=0, high=180)
        cv2.ellipse(mask, center=(blur_pos[1] + ellipse_width // 2, blur_pos[0] + ellipse_height // 2),
                    axes=(ellipse_width // 2, ellipse_height // 2),
                    angle=angle, startAngle=0, endAngle=360, color=255, thickness=-1)
        open_result = mask

        # 高斯模糊
        result = cv2.GaussianBlur(open_result, (21, 21), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

    dsize = (shape[1], shape[0])
    result_re = cv2.resize(result, dsize, interpolation=cv2.INTER_LANCZOS4)

    return result_re / 255.0
