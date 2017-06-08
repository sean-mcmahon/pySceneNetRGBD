"""
This script will read unshuffed images from LMDBs, shuffle, do some processing
and then save to new LMDBs.
Key Tasks:
- Read in images from the three LMDBs for each tar
- Append random int to keys to shuffle, same ints to same keys accross img types
- Convert the Instance images to NYU 13 labels
- Make sure Image data types compabible with Caffe (float and uint8 for label)
- Check status of images, compare to img read from file (val only)

By Sean McMahon, 7th June 2017
"""
import numpy as np
import lmdb
import os
try:
    import caffe
except ImportError:
    # could be running on hpc
    import subprocess
    cmd = 'module load caffe'
    subprocess.call(cmd, shell=True)
    import caffe
from PIL import Image  # a part of caffe module on HPC
import scenenet_pb2 as sn
import cv2

NYU_WNID_TO_CLASS = {
    '04593077': 4, '03262932': 4, '02933112': 6, '03207941': 7, '03063968': 10, '04398044': 7, '04515003': 7,
    '00017222': 7, '02964075': 10, '03246933': 10, '03904060': 10, '03018349': 6, '03786621': 4, '04225987': 7,
    '04284002': 7, '03211117': 11, '02920259': 1, '03782190': 11, '03761084': 7, '03710193': 7, '03367059': 7,
    '02747177': 7, '03063599': 7, '04599124': 7, '20000036': 10, '03085219': 7, '04255586': 7, '03165096': 1,
    '03938244': 1, '14845743': 7, '03609235': 7, '03238586': 10, '03797390': 7, '04152829': 11, '04553920': 7,
    '04608329': 10, '20000016': 4, '02883344': 7, '04590933': 4, '04466871': 7, '03168217': 4, '03490884': 7,
    '04569063': 7, '03071021': 7, '03221720': 12, '03309808': 7, '04380533': 7, '02839910': 7, '03179701': 10,
    '02823510': 7, '03376595': 4, '03891251': 4, '03438257': 7, '02686379': 7, '03488438': 7, '04118021': 5,
    '03513137': 7, '04315948': 7, '03092883': 10, '15101854': 6, '03982430': 10, '02920083': 1, '02990373': 3,
    '03346455': 12, '03452594': 7, '03612814': 7, '06415419': 7, '03025755': 7, '02777927': 12, '04546855': 12,
    '20000040': 10, '20000041': 10, '04533802': 7, '04459362': 7, '04177755': 9, '03206908': 7, '20000021': 4,
    '03624134': 7, '04186051': 7, '04152593': 11, '03643737': 7, '02676566': 7, '02789487': 6, '03237340': 6,
    '04502670': 7, '04208936': 7, '20000024': 4, '04401088': 7, '04372370': 12, '20000025': 4, '03956922': 7,
    '04379243': 10, '04447028': 7, '03147509': 7, '03640988': 7, '03916031': 7, '03906997': 7, '04190052': 6,
    '02828884': 4, '03962852': 1, '03665366': 7, '02881193': 7, '03920867': 4, '03773035': 12, '03046257': 12,
    '04516116': 7, '00266645': 7, '03665924': 7, '03261776': 7, '03991062': 7, '03908831': 7, '03759954': 7,
    '04164868': 7, '04004475': 7, '03642806': 7, '04589593': 13, '04522168': 7, '04446276': 7, '08647616': 4,
    '02808440': 7, '08266235': 10, '03467517': 7, '04256520': 9, '04337974': 7, '03990474': 7, '03116530': 6,
    '03649674': 4, '04349401': 7, '01091234': 7, '15075141': 7, '20000028': 9, '02960903': 7, '04254009': 7,
    '20000018': 4, '20000020': 4, '03676759': 11, '20000022': 4, '20000023': 4, '02946921': 7, '03957315': 7,
    '20000026': 4, '20000027': 4, '04381587': 10, '04101232': 7, '03691459': 7, '03273913': 7, '02843684': 7,
    '04183516': 7, '04587648': 13, '02815950': 3, '03653583': 6, '03525454': 7, '03405725': 6, '03636248': 7,
    '03211616': 11, '04177820': 4, '04099969': 4, '03928116': 7, '04586225': 7, '02738535': 4, '20000039': 10,
    '20000038': 10, '04476259': 7, '04009801': 11, '03909406': 12, '03002711': 7, '03085602': 11, '03233905': 6,
    '20000037': 10, '02801938': 7, '03899768': 7, '04343346': 7, '03603722': 7, '03593526': 7, '02954340': 7,
    '02694662': 7, '04209613': 7, '02951358': 7, '03115762': 9, '04038727': 6, '03005285': 7, '04559451': 7,
    '03775636': 7, '03620967': 10, '02773838': 7, '20000008': 6, '04526964': 7, '06508816': 7, '20000009': 6,
    '03379051': 7, '04062428': 7, '04074963': 7, '04047401': 7, '03881893': 13, '03959485': 7, '03391301': 7,
    '03151077': 12, '04590263': 13, '20000006': 1, '03148324': 6, '20000004': 1, '04453156': 7, '02840245': 2,
    '04591713': 7, '03050864': 7, '03727837': 5, '06277280': 11, '03365592': 5, '03876519': 8, '03179910': 7,
    '06709442': 7, '03482252': 7, '04223580': 7, '02880940': 7, '04554684': 7, '20000030': 9, '03085013': 7,
    '03169390': 7, '04192858': 7, '20000029': 9, '04331277': 4, '03452741': 7, '03485997': 7, '20000007': 1,
    '02942699': 7, '03231368': 10, '03337140': 7, '03001627': 4, '20000011': 6, '20000010': 6, '20000013': 6,
    '04603729': 10, '20000015': 4, '04548280': 12, '06410904': 2, '04398951': 10, '03693474': 9, '04330267': 7,
    '03015149': 9, '04460038': 7, '03128519': 7, '04306847': 7, '03677231': 7, '02871439': 6, '04550184': 6,
    '14974264': 7, '04344873': 9, '03636649': 7, '20000012': 6, '02876657': 7, '03325088': 7, '04253437': 7,
    '02992529': 7, '03222722': 12, '04373704': 4, '02851099': 13, '04061681': 10, '04529681': 7,
}


def img_to_datum(img, encode=False):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels, datum.height, datum.width = img.shape
    if datum.channels > 3:
        raise(Exception('Invalide number of channels. Has the image been ' +
                        'reshaped into Caffe format? shape={}'.format(img.shape)))
    if encode:
        en_img = cv2.imencode('.jpg', img)[1]
        if en_img is None:
            # have to do this check because opencv is a piece of shit
            infostr = 'Img shape {}; dtype {}'.format(img.shape, img.dtype)
            raise(Exception('image encoding failed.\n' + infostr))
        datum.data = en_img.tobytes()
        datum.encoded = True
    elif img.dtype == np.uint8:
        datum.data = img.tobytes()
        datum.encoded = False
    else:
        datum.float_data.extend(img.flat)
        datum.encoded = False
    return datum


def string_to_image(lmdb_val_raw):
    """
    Covnerts raw lmdb serialised information into a w,h,channel image
    """
    # convert to np array then check
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(lmdb_val_raw)
    if not datum.encoded:
        img = caffe.io.datum_to_array(datum)
        if datum.channels == 3:
            img = img.transpose((1, 2, 0))
            img = img[..., ::-1]
        elif datum.channels == 1:
            img = np.squeeze(img)
        else:
            err = 'Datum not single channel or RGB {} channels'.format(
                datum.channels)
            raise(Exception(err))
    return img


def checkImg(key, value):
    """
    Compares LMDB image with image read from file. Making sure they are the same.
    Can only process for the validation set.
    Check image format as well, should be float or uint8
    """
    if '/home/sean' in os.path.expanduser('~'):
        img_path = '/home/sean/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
    else:
        img_path = '/work/cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
    im_name = os.path.join(img_path, key)
    if not os.path.isfile(im_name):
        err_st = 'image filename "{}" does not exist'.format(im_name)
        raise(Exception(err_st))
    fileimg = np.asarray(Image.open(im_name))
    if type(value).__module__ == np.__name__:
        # value is np array, do comparison
        return np.array_equal(value, fileimg)
    else:
        # convert to np array then check
        datum_img = string_to_image(value)
        if not np.array_equal(datum_img, fileimg):
            print 'datum_img has  shape {}, num unique el {}'.format(
                datum_img.shape, len(np.unique(datum_img)))
            print 'file image has shape {}, num unique el {}'.format(
                fileimg.shape, len(np.unique(fileimg)))
            return False
        else:
            return True


def convert_nyu(key, value, trajectories):
    """
    Read label/instance image and convert from wordnet labels to NYU13.
    Based on convert_instance2class.py
    """
    datum = caffe.proto.caffe_pb2.Datum()
    # TODO only ParseFromString once! (done once in check and again in check)
    datum.ParseFromString(value)
    instance_img = np.squeeze(caffe.io.datum_to_array(datum))
    if not checkImg(key, instance_img):
        raise(Exception('instance_img does not match file loaded img'))

    class_img = np.zeros(instance_img.shape)

    # find view in protobuf
    key_traj = os.path.basename(os.path.abspath(os.path.join(key, os.pardir,
                                                             os.pardir)))
    (key_frame_num, _) = os.path.splitext(os.path.basename(key))
    instance_class_map = {}
    for traj in trajectories.trajectories:
        if key_traj in traj.render_path:
            # found the scene (trajectory)
            # get mapping from wordnet instances to NYU13
            # doing this for every instance image is incredibly inefficient
            # TODO creation instance_class_map before convert_nyu is called
            for instance in traj.instances:
                if instance.instance_type != sn.Instance.BACKGROUND:
                    instance_class_map[instance.instance_id] = NYU_WNID_TO_CLASS[
                        instance.semantic_wordnet_id]
            for view_id, view in enumerate(traj.views):
                if int(key_frame_num) == view.frame_num:
                    # found the frame!
                    for instance_, nyu_class in instance_class_map.items():
                        class_img[instance_img == instance_] = nyu_class
                    return np.uint8(class_img[np.newaxis, ...])
    print 'No matching instances in trajectories'
    return None


if __name__ == '__main__':
    trajectories = sn.Trajectories()
    scenenet_path = '/home/sean/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
    protobuf_path = os.path.join(scenenet_path,
                                 'pySceneNetRGBD/data/scenenet_rgbd_val.pb')
    print 'opening trajectories protobuf...'
    if not os.path.isfile(protobuf_path):
        raise(Exception('Invalid protobuf path: %s' % protobuf_path))
    with open(protobuf_path, 'rb') as f:
        trajectories.ParseFromString(f.read())
    lmdb_path = '/home/sean/hpc-cyphy/SeanMcMahon/datasets/SceneNet_RGBD/'
    lmdb_names = ['val_1000_instance', 'val_1000_rgb', 'val_1000_depth']
    img_LMDBs = [os.path.join(lmdb_path, lmdb_name)
                 for lmdb_name in lmdb_names]
    r_seed = 9421  # np.random.randint(999, 10000)
    for lmdb_name in img_LMDBs:
        print 'Creating LMDBs...'
        env = lmdb.open(lmdb_name, readonly=True)
        [path, tail] = os.path.split(lmdb_name)
        new_name = path + 'shuffle_' + tail
        new_env = lmdb.open(new_name, map_size=int(1e+10))

        np.random.seed(r_seed)  # same order of rand numers for earch img type
        num_imgs = 300 * 1000
        with env.begin() as txn, new_env.begin(write=True) as w_txn:
            cursor = txn.cursor()
            # move to start of database
            if not cursor.first():
                raise(Exception('Could locate beginning of database, could be empty'))
            print 'Looping over LMDB...'
            for count, (key, value) in enumerate(cursor):
                # check and format images
                if 'instance' in key.lower():
                    nyu13_classes = convert_nyu(key, value, trajectories)
                    datum = img_to_datum(nyu13_classes, encode=False)
                    n_value = datum.SerializeToString()
                else:
                    if not checkImg(key, value):
                        print 'Issue with: ', key, ' in ', lmdb_name
                        raise(Exception('Image in LMDB different to image in file'))
                    # n_value = makefloat(value) Going to assume imags already
                    # floats
                    n_value = value
                r_int = np.random.randint(0, num_imgs)
                n_key = '{0:0>6d}_'.format(r_int) + key
                # write to new lmdb
                if not w_txn.put(n_key.encode('ascii'), n_value):
                    # failed or key is duplicated
                    raise(Exception('Saving "{}" failed'.format(n_key)))
                if count % 250 == 0:
                    print 'saved {}/1000 to {}'.format(count,
                                                       os.path.basename(lmdb_name))
                    if count > 0:
                        print 'early bbreak'
                        break
