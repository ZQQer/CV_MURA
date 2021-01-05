import numpy as np
import cv2
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

def tsmooth(I, lambda_=0.01, sigma=3.0, sharpness=0.02, max_iter=4):
    I = im2double(I)
    x = I
    sigma_iter = sigma
    lambda_ = lambda_ / 2
    dec = 2

    for i in range(max_iter):
        wx, wy = compute_texture_weights(x, sigma_iter, sharpness)
        x = solve_linear_equation(I, wx, wy, lambda_)
        sigma_iter /= dec
        if sigma_iter < 0.5:
            sigma_iter = 0.5

    return x

def im2double(image):
    return image / 255

def compute_texture_weights(fin, sigma, sharpness):
    fx = np.diff(fin, 1, 1)
    fx = np.pad(fx, [(0,0),(0,1),(0,0)], 'constant')
    fy = np.diff(fin, 1, 0)
    fy = np.pad(fy, [(0,1),(0,0),(0,0)], 'constant')

    vareps_s = sharpness
    vareps = 0.001

    wto = np.maximum(((fx**2 + fy**2) ** 0.5).mean(axis=-1), vareps_s) ** (-1)
    fbin = lpfilter(fin, sigma)
    gfx = np.diff(fbin, 1, 1)
    gfx = np.pad(gfx, [(0,0),(0,1),(0,0)], 'constant')
    gfy = np.diff(fbin, 1, 0)
    gfy = np.pad(gfy, [(0,1),(0,0),(0,0)], 'constant')
    wtbx = np.maximum(np.abs(gfx).mean(axis=-1), vareps) ** (-1)
    wtby = np.maximum(np.abs(gfy).mean(axis=-1), vareps) ** (-1)
    retx = wtbx * wto
    rety = wtby * wto

    retx[:, -1] = 0
    rety[-1, :] = 0

    return retx, rety

# ref at https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def conv2_sep(im, sigma):
    ksize = round(5 * sigma) | 1
    g = matlab_style_gauss2D((1,ksize), sigma)
    ret = cv2.filter2D( im, -1,   g, borderType=cv2.BORDER_CONSTANT)
    ret = cv2.filter2D(ret, -1, g.T, borderType=cv2.BORDER_CONSTANT)
    return ret

def lpfilter(image, sigma):
    filtered = np.array([conv2_sep(array, sigma) for array in image.transpose((2,0,1))])
    filtered = filtered.transpose((1,2,0))
    return filtered

def solve_linear_equation(in_, wx, wy, lambda_):
    r, c, ch = in_.shape
    k = r * c
    dx = -lambda_ * wx.flatten('F')
    dy = -lambda_ * wy.flatten('F')
    B = np.stack((dx, dy))
    d = (-r, -1)
    A = spdiags(B, d, k, k)
    e = dx
    w = np.pad(dx, [(r,0)], 'constant')[:-r]
    s = dy
    n = np.pad(dy, [(1,0)], 'constant')[:-1]
    D = 1 - (e+w+s+n)
    A = A + A.T + spdiags(D, 0, k, k)

    out = np.zeros((r, c, ch))
    for i in range(ch):
        tin = in_[..., i].flatten('F')
        tout = spsolve(A, tin)
        out[..., i] += tout.reshape(c, r).T

    return out

if __name__ == '__main__':
    import argparse
    import tqdm
    import os
    from os.path import join, exists
    from PIL import Image
    # name = '../MURA_E_txt/structure_test.list'
    # a = np.genfromtxt(name, dtype=np.str, encoding='utf-8')
    # if 'gt' in name:
    #     a = [i.replace('/home/zrx/zrx/data/MURA-v1.1/valid/XR_ELBOW','./data/org') for i in a]
    #     np.savetxt(name, a, fmt='%s')
    # elif 'struc' in name:
    #     a = [i.replace('/home/zrx/zrx/data/MURA-v1.1-RTV/valid/XR_ELBOW', './data/RTV') for i in a]
    #     np.savetxt(name, a, fmt='%s')
    # else:
    #     print('wrong')

    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_', default=0.015, type=float)  ## fix
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument('--sharpness', default=0.008, type=float)
    parser.add_argument('--max_iter', default=1, type=int)
    args = parser.parse_args()

    input_imgs = np.genfromtxt('../MURA_E_txt/gt_for_visual.list', dtype=np.str, encoding='utf-8')

    for i in tqdm.trange(len(input_imgs)):
        image = np.array(Image.open('.'+input_imgs[i]).convert('RGB'))
        print(image.shape)

        smoothed = tsmooth(image, args.lambda_, args.sigma, args.sharpness, args.max_iter)
        d = '../data/RTV/'+input_imgs[i].split('/')[-3]+'/'+input_imgs[i].split('/')[-2] +'/'#+input_imgs[i].split('/')[-1]
        if not exists(d):
            os.makedirs(d)
        Image.fromarray((smoothed * 256).astype('uint8')).save(d+input_imgs[i].split('/')[-1])

