import numpy as np
import re
import time
import random
from math import *
from sklearn.neighbors import NearestNeighbors
from utils import *


def read_file_original(file_path):
    a = []
    with open(file_path) as f:
        content = f.readlines()
        for line in content:
            x = float(re.split('\s+', line)[0])
            y = float(re.split('\s+', line)[1])
            z = float(re.split('\s+', line)[2])

            b = np.array([x,y,z])
            a.append(b)

    data = np.array(a)
    return data


def read_file_deformed(file_path):
    a = []
    with open(file_path) as f:
        content = f.readlines()
        for line in content:
            x = float(re.split('\s+', line)[0])
            y = float(re.split('\s+', line)[1])
            z = float(re.split('\s+', line)[2])

            nx = float(re.split('\s+', line)[3])
            ny = float(re.split('\s+', line)[4])
            nz = float(re.split('\s+', line)[5])

            b = np.array([x,y,z,nx,ny,nz])
            a.append(b)


    data = np.array(a)
    return data

def generateTransformByEuler(a, b, c, x, y, z):
    a = a / 180 * 3.14159
    b = b / 180 * 3.14159
    c = c / 180 * 3.14159
    R = np.array([[cos(b)*cos(c), sin(a)*sin(b)*cos(c) - cos(a)*sin(c), cos(a)*sin(b)*cos(c) + sin(a)*sin(c)],
                  [cos(b)*sin(c), cos(a)*cos(c) + sin(a)*sin(b)*sin(c), cos(a)*sin(b)*sin(c) - cos(c)*sin(a)],
                  [-sin(b), cos(b)*sin(a), cos(a)*cos(b)]])
    R1 = np.array([[1, -c, b],
                   [c, 1, -a],
                   [-b, a, 1]])
    t = np.array([x, y, z])
    T = np.identity(4)
    T[0 : 3, 0 : 3] = R
    T[0 : 3, 3] = t
    return T

def generatePerturbation(theta, translation):
    n = np.random.random((1, 3))
    n = n / np.linalg.norm(n)
    theta = theta / 180 * 3.14159
    if translation != 0:
        t = np.random.random((1, 3))
        t = t / np.linalg.norm(t) / sqrt(1 / translation)
    else:
        t = np.zeros((1, 3))
    antisymmetric = np.array([[0, -n[0, 2], n[0, 1]],
                              [n[0, 2], 0, -n[0, 0]],
                              [-n[0, 1], n[0, 0], 0]])
    R = cos(theta) * np.identity(3) + (1 - cos(theta)) * np.transpose(n).dot(n) + sin(theta) * antisymmetric
    Tg = np.identity(4)
    Tg[0: 3, 0: 3] = R
    Tg[0: 3, 3] = t
    return Tg


def eval_err(Tr, Tg):
    detT = Tr.dot(np.linalg.inv(Tg))
    err_t = sqrt((detT[0, 3]**2 + detT[1, 3]**2 + detT[2, 3]**2))
    #print ("Tr:",Tr)
    #print ("Tg:",Tg)
    try:
        err_R = acos((detT[0 ,0] + detT[1, 1] + detT[2, 2] - 1) / 2)
    except:
        err_R = 0
    return (err_t, err_R)


def randomDownsample(cloud, num):

    assert num < cloud.shape[0]

    N_cloud = cloud.shape[0]

    sample = np.array(random.sample(range(1,N_cloud) ,num))
    cloud = cloud[sample,: ]

    return cloud


def cloudTransform(T, cloud):

    N_src = cloud.shape[0]

    cloudHomo = np.ones((N_src, 4))
    cloudHomo[ : , :3] = cloud
    clouddstHomo = np.dot(T, cloudHomo.T)

    return clouddstHomo[ :3 , :].T



def calculateTransform(src, dst):

    assert  src.shape == dst.shape

    centroidA = np.mean(src, axis=0)
    centroidB = np.mean(dst, axis=0)

    srcFromC = src - centroidA
    dstFromC = dst - centroidB
    H = np.dot(srcFromC.T, dstFromC)

    U, S, V = np.linalg.svd(H)
    R = np.dot(V.T, U.T)
    if np.linalg.det(R) < 0:
        V[2,:] *= -1
        R = np.dot(V.T, U.T)
    t = centroidB.T - np.dot(R, centroidA.T)

    T = np.identity(4)
    T[ :3, :3] = R
    T[ :3 , 3] = t

    return T


def matchPoints_plane(src, dst):
    pass


def matchPoints_proj(src, dst):

    src2D = src[ : , : 2]
    dst2D = dst[ : , : 2]

    distances, indices = matchPoints_point(src2D, dst2D)

    return distances, indices




def matchPoints_point(src, dst):

    assert  src.shape[1] == dst.shape[1]

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()


def estimateNormals(p, k=4):
    '''
    :param p: point cloud m*3
    :param k: Knn
    :return: normals with ambiguous orientaion (PCA solution problem)
    '''
    knn = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(p)
    _, index = knn.kneighbors(p)

    m = p.shape[0]
    normals = np.zeros((m,3))
    for i in range(m):
        nn = p[index[i,1:]] # exclude self in nn
        c = np.cov(nn.T) # covariance

        # print ("c", c)
        w,v = np.linalg.eig(c)
        normals[i] = v[:,np.argmin(w)]

    pWithNormal = np.zeros((m, 6))
    for i in range(m):
        pWithNormal[i, 0] = p[i, 0]
        pWithNormal[i, 1] = p[i, 1]
        pWithNormal[i, 2] = p[i, 2]
        pWithNormal[i, 3] = normals[i, 0]
        pWithNormal[i, 4] = normals[i, 1]
        pWithNormal[i, 5] = normals[i, 2]
    return pWithNormal


'''def pointcloud2image(point_cloud):
    x_size = 640
    y_size = 640
    x_range = 60.0
    y_range = 60.0
    grid_size = np.array([2 * x_range / x_size, 2 * y_range / y_size])
    image_size = np.array([x_size, y_size])
    # [0, 2*range)
    shifted_coord = point_cloud[:, :2] + np.array([x_range, y_range])
    # image index
    index = np.floor(shifted_coord / grid_size).astype(np.int)
    # choose illegal index
    bound_x = np.logical_and(index[:, 0] >= 0, index[:, 0] < image_size[0])
    bound_y = np.logical_and(index[:, 1] >= 0, index[:, 1] < image_size[1])
    bound_box = np.logical_and(bound_x, bound_y)
    index = index[bound_box]
    # show image
    image = np.zeros((640, 640), dtype=np.uint8)
    image[index[:, 0], index[:, 1]] = 255
    res = Image.fromarray(image)
    # rgb = Image.merge('RGB', (res, res, res))
    res.show()'''
