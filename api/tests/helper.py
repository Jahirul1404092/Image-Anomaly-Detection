import functools
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import os
import re
from unittest.mock import Mock
from functools import wraps

from skimage.io import imsave
import numpy as np

from tests.helpers.dataset import generate_random_anomaly_image

__all__ = [
    'sub_test',
    'generateRandomImage',
    'createLetters',
    'createShapes',
    'merge_dicts',
    'isDicInDic',
    'verifier_mock'
]

# parameterized test tool
def sub_test(param_list):
    """Decorates a test case to run it as a set of subtests."""

    def decorator(f):

        @functools.wraps(f)
        def wrapped(self):
            for param in param_list:
                with self.subTest(**param):
                    f(self, **param)

        return wrapped

    return decorator

# generates random letter images
def generateRandomImage(path, bad=False):
    img = Image.new('RGB', (300, 300), 
                    (random.randint(120, 255), 
                     random.randint(120, 255), 
                     random.randint(120, 255)))
    draw = ImageDraw.Draw(img)
    for i in range(random.randint(15, 25)):
        ttf = random.choice(list(Path('/usr/share/fonts').glob('**/*.ttf')))
        font = ImageFont.truetype(str(ttf), random.randint(28, 72))
        l_letter = random.choices(str(ttf.stem), k=5)
        if bad:
            l_letter[random.randint(0,4)] = '#'
        text = ''.join(l_letter)
        # textWidth_, textHeight_ = draw.textsize(text,font=font) # depricated 
        _, _, textWidth, textHeight = draw.textbbox((0, 0), text, font=font)
        textTopLeft = (random.randint(-20, max(300 - textWidth, -20)), 
                       random.randint(-20, max(300 - textHeight, -20)))
        draw.text(textTopLeft, text, 
                  fill=(random.randint(10, 100), 
                        random.randint(10, 100), 
                        random.randint(10, 100)), 
                  font=font)
    img.save(path)
    return np.array(img)

def createLetters(imgpath, trainsize, prefix=''):
    hostdir = os.environ.get('HOST_DATA_DIR', '/test')
    basedir = os.environ.get('BASE_DATA_DIR', '/tmp')
    l_good = []
    for i in range(0, trainsize):
        path = f"{imgpath}/{prefix}{i + 1:03}.png"
        generateRandomImage(path)
        l_good.append(re.sub(f"^{basedir}", hostdir, path))
    l_bad = []
    l_mask = []
    maskpath = imgpath + '/mask'
    Path(maskpath).mkdir()
    for i in range(trainsize, trainsize + int(trainsize * 0.2)):
        path = f"{imgpath}/{prefix}{i + 1:03}.png"
        image = generateRandomImage(path, bad=True)
        l_bad.append(re.sub(f"^{basedir}", hostdir, path))
        path = f"{maskpath}/{prefix}{i + 1:03}.png"
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[image[..., 0] < 255] = 255
        mask[image[..., 1] < 255] = 255
        mask[image[..., 2] < 255] = 255
        imsave(path, mask, check_contrast=False)
        l_mask.append(re.sub(f"^{basedir}", hostdir, path))
    return l_good, l_bad, l_mask

def createShapes(imgpath, trainsize, prefix=''):
    hostdir = os.environ.get('HOST_DATA_DIR', '/test')
    basedir = os.environ.get('BASE_DATA_DIR', '/tmp')
    l_good = []
    for i in range(0, trainsize):
        path = f"{imgpath}/{prefix}{i + 1:03}.png"
        result = generate_random_anomaly_image(
            320, 320, ['triangle', 'rectangle', 'hexagon'], generate_mask=False
        )
        image = result['image']
        imsave(path, image, check_contrast=False)
        l_good.append(re.sub(f"^{basedir}", hostdir, path))
    l_bad = []
    l_mask = []
    maskpath = imgpath + '/mask'
    Path(maskpath).mkdir()
    for i in range(trainsize, trainsize + int(trainsize * 0.2)):
        result = generate_random_anomaly_image(
            320, 320, ['triangle', 'rectangle', 'hexagon', 'star'], generate_mask=True
        )
        path = f"{imgpath}/{prefix}{i + 1:03}.png"
        image = result['image']
        imsave(path, image, check_contrast=False)
        l_bad.append(re.sub(f"^{basedir}", hostdir, path))
        path = f"{maskpath}/{prefix}{i + 1:03}.png"
        image = result['mask']
        imsave(path, image, check_contrast=False)
        l_mask.append(re.sub(f"^{basedir}", hostdir, path))
    return l_good, l_bad, l_mask

def merge_dicts(tgt, enhancer):
    for key, val in enhancer.items():
        if key not in tgt:
            tgt[key] = val
            continue
        if isinstance(val, dict):
            merge_dicts(tgt[key], val)
        else:
            tgt[key] = val
    return tgt

def isDicInDic(comp, tgt):
    for key, val in comp.items():
        if key not in tgt:
            return False
        if isinstance(val, dict):
            return isDicInDic(val, tgt[key])
        else:
            if tgt[key] != val:
                return False
    return True

def verifier_mock(f):
    """Mock for license verification."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper
