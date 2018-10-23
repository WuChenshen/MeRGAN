
from os import listdir
import numpy as np
import scipy.misc
import time
import pdb
Label={'bedroom':0,
       'kitchen':1,
       'dining_room':2,
       'conference_room':3,
       'living_room':4,
       'bridge':5,
       'tower':6,
       'classroom':7,
       'church_outdoor':8,
       'restaurant':9}

def make_generator(path, classes, batch_size, image_size, pharse='train'):
    epoch_count = [1]
    image_list = []
    for sub_class in classes:
        sub_class_path = path + '/'+ sub_class + '/'+ pharse
        sub_class_image = listdir(sub_class_path)        
        image_list.extend([sub_class_path + '/' + i for i in sub_class_image])

    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='float32')
        labels = np.zeros((batch_size,), dtype='int32')
        files = range(len(image_list))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            #image = scipy.misc.imread("{}/{}.png".format(path, str(i+1).zfill(len(str(n_files)))))
            image = scipy.misc.imread("{}".format(image_list[i]))
            label = Label[image_list[i].split('/')[4]]
            image = scipy.misc.imresize(image,(image_size,image_size))
            images[n % batch_size] = image.transpose(2,0,1) / 255.
            labels[n % batch_size] = label
            if n > 0 and n % batch_size == 0:
                yield (images,labels)    
    
    return get_epoch

def load(batch_size, classes, data_dir='/datatmp/dataset/LSUN_10_100000_old',image_size = 64):
    return (
        make_generator(data_dir, classes, batch_size, image_size, pharse='train'),
        make_generator(data_dir, classes, batch_size, image_size, pharse='val')
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()

