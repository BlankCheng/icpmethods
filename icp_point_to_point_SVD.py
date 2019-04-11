from utils import *

def icpPointToPointSVD(src, dst, init = None, maxIterations=20, fitnessThres=1e-4, tranformThres=1e-4):

    assert  src.shape[1] == dst.shape[1]
    N_src = src.shape[0]
    N_dst = dst.shape[0]

    srcHomo = np.ones((N_src, 4))
    dstHomo = np.ones((N_dst, 4))
    srcHomo[ : , :3] = src
    dstHomo[ : , :3] = dst
    srcHomo = srcHomo.T
    dstHomo = dstHomo.T
    if init:
        srcHomo = np.dot(init, srcHomo)

    prev_err = 0
    Tr = np.identity(4)

    for n in range(maxIterations):

        distances, indices = matchPoints_point(srcHomo[:3 , :].T, dstHomo[ :3 , :].T)

        T = calculateTransform(srcHomo[ :3 , :].T, dstHomo[ :3 , indices].T)

        Tr = np.dot(T, Tr)
        srcHomo = np.dot(T, srcHomo)
        cur_err = np.mean(distances)
        if cur_err < fitnessThres:
            print ("Less than fitnessThres. Done!")
            break
        if np.abs(cur_err - prev_err) < tranformThres:
            print ("Less than transformThres. Done!")
            break
        prev_err = cur_err

    return Tr, cur_err, n + 1


def icpPointToPlane(src, dst, init = None, maxIterations=20, fitnessThres=1e-4, tranformThres=1e-4):
    pass

if __name__ == '__main__':
    Tg = generatePerturbation(theta=30, translation=0.3)
    cloudRead = np.random.rand(20000, 3)
    cloudReference = cloudTransform(Tg, cloudRead)
    cloudRead = randomDownsample(cloudRead, 10000)
    cloudReference = randomDownsample(cloudReference, 10000)
    t1 = time.clock()
    Tr, _, n = icpPointToPointSVD(cloudRead, cloudReference, init = None, maxIterations=100, fitnessThres=1e-4, tranformThres=1e-4)
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





