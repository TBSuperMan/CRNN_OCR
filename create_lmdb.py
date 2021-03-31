import lmdb  # install lmdb by "pip install lmdb"
import re
import six
from PIL import Image
import torchvision.transforms as transforms

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def _is_difficult(word):
    assert isinstance(word, str)
    return not re.match('^[\w]+$', word)


def createDataset(data_dir, outputPath, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    # assert(len(imagePathList) == len(labelList))
    # nSamples = len(imagePathList)

    data_env = lmdb.open(data_dir)
    env = lmdb.open(outputPath, map_size=8589934592)
    with data_env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))

    cache = {}
    cnt = 1
    for i in range(nSamples):
        with data_env.begin(write=False) as txn:
            img_key = b'image-%09d' % (i+1)
            imgbuf = txn.get(img_key)
            label_key = b'label-%09d' % (i+1)
            label = txn.get(label_key)

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('RGB')
        except IOError:
            print('Corrupted image for %d' % i)
            return
        toTensor=transforms.ToTensor()
        img=toTensor(img)
        print(i,img.shape)
        c,h,w=img.shape
        if h > 2*w:
            continue

        cache[imageKey] = imgbuf
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == "__main__":
    """
    gt_file: the annotation of the dataset with the format:
    image_path_1 label_1
    image_path_2 label_2
    ...
  
    data_dir: the root dir of the images. i.e. data_dir + image_path_1 is the path of the image_1
    lmdb_output_path: the path of the generated LMDB file
    """
    data_dir ="D:/Document/DataSet/DataSet/Chinese_data/MTWI_test/"
    lmdb_output_path ="D:/Document/DataSet/DataSet/Chinese_data/MTWI_test_filter/"
    # data_dir ="/home/gmn/datasets/MTWI_train/"
    # lmdb_output_path ="/home/gmn/datasets/MTWI_train_filter/"
    # gt_file = os.path.join(data_dir, 'Synth_train.txt')

    # with open(gt_file, 'r') as f:
    #     lines = [line.strip('\n') for line in f.readlines()]

    # imagePathList, labelList, embedList = [], [], []
    # for i, line in enumerate(lines):
    #     splits = line.split(' ')
    #     image_name = splits[0]
    #     gt_text = splits[1]
    #     imagePathList.append(os.path.join(data_dir, image_name))
    #     labelList.append(gt_text)
    createDataset(data_dir, lmdb_output_path)