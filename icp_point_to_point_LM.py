from utils import *

def icpPointToPointLM(source_points, dest_points,initial, maxIterations,  fitnessThres=1e-4, tranformThres=1e-4):
    """
    Point to point matching using Gauss-Newton

    source_points:  nx3 matrix of n 3D points
    dest_points: nx3 matrix of n 3D points, which have been obtained by some rigid deformation of 'source_points'
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
        for i in range (0,dest_points.shape[0]-1):

            #print dest_points[i][3],dest_points[i][4],dest_points[i][5]
            dx = dest_points_tmp[i][0]
            dy = dest_points_tmp[i][1]
            dz = dest_points_tmp[i][2]

            sx = source_points[i][0]
            sy = source_points[i][1]
            sz = source_points[i][2]

            alpha = initial[0][0]
            beta = initial[1][0]
            gamma = initial[2][0]
            tx = initial[3][0]
            ty = initial[4][0]
            tz = initial[5][0]
            #print alpha

            a1 = (-2*beta*sx*sy) - (2*gamma*sx*sz) + (2*alpha*((sy*sy) + (sz*sz))) + (2*((sz*dy) - (sy*dz))) + 2*((sy*tz) - (sz*ty))
            a2 = (-2*alpha*sx*sy) - (2*gamma*sy*sz) + (2*beta*((sx*sx) + (sz*sz))) + (2*((sx*dz) - (sz*dx))) + 2*((sz*tx) - (sx*tz))
            a3 = (-2*alpha*sx*sz) - (2*beta*sy*sz) + (2*gamma*((sx*sx) + (sy*sy))) + (2*((sy*dx) - (sx*dy))) + 2*((sx*ty) - (sy*tx))
            a4 = 2*(sx - (gamma*sy) + (beta*sz) +tx -dx)
            a5 = 2*(sy - (alpha*sz) + (gamma*sx) +ty -dy)
            a6 = 2*(sz - (beta*sx) + (alpha*sy) +tz -dz)

            _residual = (a4*a4/4)+(a5*a5/4)+(a6*a6/4)

            _J = np.array([a1, a2, a3, a4, a5, a6])
            _e = np.array([_residual])

            J.append(_J)
            e.append(_e)

        jacobian = np.array(J)
        residual = np.array(e)

        update = -np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(jacobian),jacobian)),np.transpose(jacobian)),residual)

        #print update, initial

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
    Tg = generatePerturbation(theta=1, translation=0.1)
    print (Tg)
    cloudRead = np.random.rand(200, 3)
    cloudReference = cloudTransform(Tg, cloudRead)
    #cloudRead = randomDownsample(cloudRead, 10000)
    #cloudReference = randomDownsample(cloudReference, 10000)
    initial = np.array([[0.1], [0.1], [0.1], [0.01], [0.01], [0.01]])
    #cloudReferenceWithNormal = estimateNormals(cloudReference, k=4)
    t1 = time.clock()
    Tr, _, n = icpPointToPointLM(cloudRead, cloudReference, initial = initial, maxIterations=1000, fitnessThres=1e-4, tranformThres=1e-4)

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

