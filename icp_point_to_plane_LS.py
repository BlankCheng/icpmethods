# -*- coding: utf-8 -*-
from utils import *
import transformations as transform

def icpPointToPlaneLS(source_points, dest_points, maxIterations, fitnessThres=1e-4, tranformThres=1e-4):
    """
    Point to plane matching using least squares

    source_points:  nx3 matrix of n 3D points
    dest_points: nx6 matrix of n 3D points + 3 normal vectors, which have been obtained by some rigid deformation of 'source_points'
    """


    Tr = np.identity(4)
    prev_err = 0

    for iter in range(maxIterations):
        # match points
        distances, indices = matchPoints_point(source_points, dest_points[ : , : 3])
        print (indices)
        #distances, indices = matchPoints_proj(source_points, dest_points[ : , : 3 ])
        #print (indices)
        dest_points_tmp = dest_points[indices, :]
        # calculate transform
        A = []
        b = []

        for i in range (0,dest_points[indices, : ].shape[0]-1):

            dx = dest_points_tmp[i][0]
            dy = dest_points_tmp[i][1]
            dz = dest_points_tmp[i][2]
            nx = dest_points_tmp[i][3]
            ny = dest_points_tmp[i][4]
            nz = dest_points_tmp[i][5]

            sx = source_points[i][0]
            sy = source_points[i][1]
            sz = source_points[i][2]

            _a1 = (nz*sy) - (ny*sz)
            _a2 = (nx*sz) - (nz*sx)
            _a3 = (ny*sx) - (nx*sy)

            # 和论文里方法一样，直接简化到了最后一步
            _a = np.array([_a1, _a2, _a3, nx, ny, nz])
            # b为s到d切面的距离
            _b = (nx*dx) + (ny*dy) + (nz*dz) - (nx*sx) - (ny*sy) - (nz*sz)
            A.append(_a)
            b.append(_b)


        # A1: n * 6, A_: 6 * n, b1: n
        A1 = np.array(A)
        b1 = np.array(b)
        A_ = np.linalg.pinv(A1)
        tr = np.dot(A_,b) # 六元欧拉向量


        T = transform.euler_matrix(tr[0],tr[1],tr[2]) # 这里R是4 * 4的T
        T[0,3] = tr[3]
        T[1,3] = tr[4]
        T[2,3] = tr[5]


        # do transformation
        source_transformed = []

        for i in range (0,dest_points.shape[0]):
            ss = np.array([(source_points[i][0]),(source_points[i][1]),(source_points[i][2]),(1)])
            p = np.dot(T, ss)
            source_transformed.append(p[:3])

        source_points = np.array(source_transformed)

        Tr = np.dot(T, Tr)
        cur_err = np.mean(distances)
        if cur_err < fitnessThres:
            print ("Less than fitnessThres. Done!")
            break
        if np.abs(cur_err - prev_err) < tranformThres:
            print ("Less than transformThres. Done!")
            break
        prev_err = cur_err

    return Tr, cur_err, iter + 1

if __name__ == "__main__":
    Tg = generatePerturbation(theta=30, translation=0.3)
    cloudRead = np.random.rand(20000, 3)
    cloudReference = cloudTransform(Tg, cloudRead)
    #cloudRead = randomDownsample(cloudRead, 10000)
    #cloudReference = randomDownsample(cloudReference, 10000)
    cloudReferenceWithNormal = estimateNormals(cloudReference, k=4)
    t1 = time.clock()
    Tr, _, n = icpPointToPlaneLS(cloudRead, cloudReferenceWithNormal, maxIterations=100, fitnessThres=1e-4, tranformThres=1e-4)
    err_t , err_R = eval_err(Tr, Tg)
    print ("Time Consumed: ", time.clock() - t1)
    print ("Transformation: ")
    print (Tr)
    print ("Tranlation Error: ", end='')
    print (err_t)
    print ("Rotation Error: ", end='')
    print (err_R)
    print ("Iteration Times: ", end='')
    print (n)


