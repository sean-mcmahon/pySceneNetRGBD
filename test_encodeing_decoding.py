
import numpy as np
import cv2
import os
from PIL import Image

if __name__ == '__main__':
    img_name = '/home/sean/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/val/0/223/photo/0.jpg'
    pil = np.array(Image.open(img_name))
    cv = cv2.imread(img_name, 1)

    arr = np.random.randint(0, 255, size=(5, 5), dtype=np.uint8)
    bytten = arr.astype(np.uint8).tobytes()
    srtinged = np.fromstring(bytten, dtype=np.uint8).reshape(arr.shape)
    print 'test case equal? ',  np.array_equal(arr, srtinged)

    enc_img = cv2.imencode('.png', cv)[1]
    bytearr = enc_img.tobytes()
    np_bytes = np.fromstring(bytearr, dtype=np.uint8)
    print 'encoded byte image arrays equal? ', np.array_equal(
        enc_img, np_bytes.reshape(enc_img.shape))
    dec_img = cv2.imdecode(np_bytes.reshape(enc_img.shape), 1)

    eq = np.array_equal(cv, dec_img)
    print 'decoded images equal? ', eq
    print 'cv img  shape: ', cv.shape
    print 'dec_img shape: ', dec_img.shape
    print 'num different pixels ', np.sum(np.setdiff1d(cv.flatten(),
                                                       dec_img.flatten()))
    print np.setdiff1d(cv.flatten(), dec_img.flatten())
    where_not_eq = np.where(cv.flatten() != dec_img.flatten())[0]
    np.isclose(cv, dec_img)
    import pdb
    pdb.set_trace()
    # cv2.imshow('cv', cv)
    # cv2.imshow('dec', dec_img)
    # cv2.waitKey()
