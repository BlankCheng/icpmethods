from utils import *

def icpPointToPlaneLM(source_points, dest_points,initial , maxIterations, fitnessThres=1e-4, tranformThres=1e-4):
    """
    Point to plane matching using Gauss Newton

    source_points:  nx3 matrix of n 3D points
    dest_points: nx6 matrix of n 3D points + 3 normal vectors, which have been obtained by some rigid deformation of 'source_points'
    initial: 1x6 matrix, denoting alpha, beta, gamma (the Euler angles for rotation and tx, ty, tz (the translation along three axis).
                this is the initial estimate of the transformation between 'source_points' and 'dest_points'
    loop: start with zero, to keep track of the number of times it loops, just a very crude way to control the recursion

    """

    prev_err = 0
    for iter in range(maxIterations):
        # match points
        distances, indices = matchPoints_point(source_points, dest_points[ : , : 3])
        print (iter)
        #distances, indices = matchPoints_proj(source_points, dest_points[ : , : 3 ])
        #print (indices)
        dest_points_tmp = dest_points[indices, :]

        J = []
        e = []

        for i in range (0,dest_points_tmp.shape[0]-1):

            #print dest_points[i][3],dest_points[i][4],dest_points[i][5]
            dx = dest_points_tmp[i][0]
            dy = dest_points_tmp[i][1]
            dz = dest_points_tmp[i][2]
            nx = dest_points_tmp[i][3]
            ny = dest_points_tmp[i][4]
            nz = dest_points_tmp[i][5]

            sx = source_points[i][0]
            sy = source_points[i][1]
            sz = source_points[i][2]

            alpha = initial[0][0]
            beta = initial[1][0]
            gamma = initial[2][0]
            tx = initial[3][0]
            ty = initial[4][0]
            tz = initial[5][0]

            a1 = (nz*sy) - (ny*sz)
            a2 = (nx*sz) - (nz*sx)
            a3 = (ny*sx) - (nx*sy)
            a4 = nx
            a5 = ny
            a6 = nz

            _residual = (alpha*a1) + (beta*a2) + (gamma*a3) + (nx*tx) + (ny*ty) + (nz*tz) - (((nx*dx) + (ny*dy) + (nz*dz)) - ((nx*sx) + (ny*sy) + (nz*sz)))

            _J = np.array([a1, a2, a3, a4, a5, a6])
            _e = np.array([_residual])

        J.append(_J)
        e.append(_e)

        jacobian = np.array(J)
        residual = np.array(e)

        update = -np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(jacobian),jacobian)),np.transpose(jacobian)),residual)


        initial = initial + update
        cur_err = np.mean(distances)
        if cur_err < fitnessThres:
            print ("Less than fitnessThres. Done!")
            break
        if np.abs(cur_err - prev_err) < tranformThres:
            print ("Less than transformThres. Done!")
            break

    initial = initial.flatten()
    Tr = generateTransformByEuler(initial[0], initial[1], initial[2], initial[3], initial[4], initial[5])
    return Tr, cur_err, iter + 1





if __name__ == "__main__":
    Tg = generatePerturbation(theta=30, translation=0.3)
    print (Tg)
    cloudRead = np.random.rand(200, 3)
    cloudReference = cloudTransform(Tg, cloudRead)
    #cloudRead = randomDownsample(cloudRead, 10000)
    #cloudReference = randomDownsample(cloudReference, 10000)
    initial = np.array([[0.1], [0.1], [0.1], [0.01], [0.01], [0.01]])
    cloudReferenceWithNormal = estimateNormals(cloudReference, k=4)
    t1 = time.clock()
    Tr, _, n = icpPointToPlaneLM(cloudRead, cloudReferenceWithNormal, initial = initial, maxIterations=1000, fitnessThres=1e-4, tranformThres=1e-4)

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
