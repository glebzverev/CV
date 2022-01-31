import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
from numpy.linalg import inv
from random import randint

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=100):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    img_2 = rgb2gray(img)
    des = ORB(n_keypoints = n_keypoints)
    des.detect_and_extract(img_2)
    keypoints = des.keypoints
    descriptors = des.descriptors

    return keypoints, descriptors

def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """
    matrix = np.zeros((3, 3))
    n,_=points.shape
    Cx = points[:,0].sum()/n
    Cy = points[:,1].sum()/n
    S = 0
    sum=0
    center = (points - np.array([Cx,Cy]))
    for r in center:
      sum += (r[0]**2+r[1]**2)**(0.5)
    if sum == 0:
      N = 0
    else:
      N = 2**(0.5) * n * (sum)**(-1)
    matrix[0,0] = N
    matrix[1,1] = N
    matrix[2,2] = 1
    matrix[0,2] = -N*Cx
    matrix[1,2] = -N*Cy
    points_plus = np.insert(points, points.shape[1], 1, axis=1) 
    new_p = matrix.dot(points_plus.T).T
    return matrix, np.delete(new_p,2,axis=1)

def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """
    N,_ = src_keypoints.shape
    M, src = center_and_normalize_points(src_keypoints) # M src
    M_, dest = center_and_normalize_points(dest_keypoints) # M' dest
    A=np.array([])
    for j in range(N):
        Ax = np.array([[-src[j][0],-src[j][1],-1,0, 0,0,dest[j][0]*src[j][0],dest[j][0]*src[j][1],dest[j][0]]]).reshape([1,9])
        Ay = np.array([[0, 0,0,-src[j][0],-src[j][1],-1, dest[j][1]*src[j][0],dest[j][1]*src[j][1],dest[j][1]]]).reshape([1,9])
        A = np.append(A,Ax)
        A = np.append(A,Ay)
    A = A.reshape([2*N, 9])
    H = np.array(np.linalg.svd(A)[2][8]).reshape([3,3])
    H_ = np.linalg.inv(M_).dot(H).dot(M)
    return H_


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=1000, residual_threshold=5, return_matches=False):
    def rand(match):
        N=match.shape[0]
        r=[]
        points = []
        for i in range(4):
            a = randint(0,N-1)
            while ((a in r)):
                a = randint(0,N-1)
            r.append(a)
        for k in r:
            points.append(match[k])
        return points

    inlayer = 0
    min=100
    match = match_descriptors(src_descriptors, dest_descriptors)
    
    N, _  = src_keypoints.shape
    inlayer = 0
    best_points = []
    for i in range(max_trials):
        points = np.array(rand(match)).reshape([4,2])
        src = src_keypoints[points[:,0]]
        dest = dest_keypoints[points[:,1]]
        H_ = find_homography(src, dest)

        in_match = 0 
        for k in range(4):
            new = H_.dot(np.array([src[k][0],src[k][1],1]).T)[:2]
            if (np.sqrt(np.square(new-dest[k]).sum())<residual_threshold):
                in_match+=1
        if in_match > inlayer:
            inlayer = in_match
            best_points = np.array(points)
    
    src=[]
    dest=[]

    if len(best_points)==0:
        return None, src_keypoints
    else:
        for j in best_points:
            src = np.append(src, src_keypoints[j[0]])
            dest = np.append(dest, dest_keypoints[j[1]])
        src = np.array(src).reshape(4,2)
        dest = np.array(dest).reshape(4,2)
        H_ = find_homography(src, dest)

        new = H_.dot(np.array([src[k][0],src[k][1],1]).T)[:2]
        tr = DEFAULT_TRANSFORM(H_)

        if return_matches:
            return tr, best_points
        else:
            return tr


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    a = len(forward_transforms) + 1
    b = (len(forward_transforms) - 1) // 2

    res = [None] * a
    res[center_index] = DEFAULT_TRANSFORM()

    for i in range(b+1, a):
        res[i] = res[i-1] + DEFAULT_TRANSFORM(inv(forward_transforms[i-1].params))

    for i in range(b , 0, -1):
        res[i-1] = tuple(res[i]) + tuple(forward_transforms[i-1])

    return tuple(res)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
        """
    N = len(simple_center_warps)
    corners = get_corners(image_collection, simple_center_warps)
    min_cor, max_cor = get_min_max_coords(list(corners))
    A = np.array([[1,0,-min_cor[1]],[0,1,-min_cor[0]],[0,0,1]])
    mod = DEFAULT_TRANSFORM(A)
    final_center = []
    for i in range(N):
        final_center.append(simple_center_warps[i]+mod)
    left = int(max_cor[0] - min_cor[0]) + 1
    right = int(max_cor[1] - min_cor[1]) + 1
    return final_center, (right, left)


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    warped_image = warp(image, rotate_transform_matrix(transform).inverse, output_shape)
    bool_array = np.ones(image.shape[:-1], dtype = np.bool8)
    warped_mask = warp(bool_array, rotate_transform_matrix(transform).inverse, output_shape)
    print(warped_image.shape, warped_mask.shape)
    return warped_image, warped_mask


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    res = np.zeros(output_shape + (3,))

    for i in range(len(image_collection) - 1, -1, -1):
        i_w, m_w = warp_image(image_collection[i], final_center_warps[i], output_shape)
        res = np.where(np.tile(m_w[..., None], (1, 1, 3)), i_w, res)

    return np.clip(res, 0, 1)


def get_gaussian_pyramid(image, n_layers, sigma):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    gaus = []
    gaus.append(image)
    for i in range(1, n_layers):
        gaus.append(gaussian(gaus[i - 1], sigma))
    return gaus


def get_laplacian_pyramid(image, n_layers, sigma):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    gaus = get_gaussian_pyramid(image, n_layers, sigma)
    lap = []
    for i in range(1, n_layers):
        lap[i] = gaus[i-1] - gaus[i]
    lap[n_layers-1] = gaus[n_layers-1]
    return lap


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    res = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        res.append(img)

    return res


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers, image_sigma, merge_sigma):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    col = increase_contrast(image_collection)
    corners = list(get_corners(im_col, final_center_warps))
    warped_images = [warp_image(img, warp, output_shape) for img, warp in zip(im_col, final_center_warps)]
    res = np.zeros(output_shape + (3,))
    res += warped_images[0][0]
    n = len(col)
    for i in range(1, n):
        right_min = int((corners[i][0, 0] + corners[i][1, 0]) / 2)
        left_max = int((corners[i - 1][2, 0] + corners[i - 1][3, 0]) / 2)

        resp = get_laplacian_pyramid(res, n_layers, image_sigma)
        warped_images_pyramid = get_laplacian_pyramid(warped_images[i][0], n_layers, image_sigma)

        mask=np.zeros(output_shape)[:, (left_max + right_min) // 2:] = 1
        pm = get_gaussian_pyramid(mask, n_layers, merge_sigma)

        res = np.zeros(res.shape)
        for j in range(n_layers):
            for k in range(0, 3):
                res[..., k] += resp[j][..., k] * np.subtract(1, pm[j])
                res[..., k] += warped_images_pyramid[j][..., k] * pm[j]

    return np.clip(res, 0, 1)

def cylindrical_inverse_map(coords, h, w, scale):
    pass

def warp_cylindrical(img, scale=None, crop=True):
    pass