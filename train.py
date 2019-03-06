import argparse
import time
import os
import tensorflow as tf
import inception_resnet_v2
from tensorflow.python.ops import data_flow_ops
import sys
import numpy as np
from six.moves import xrange
import itertools
def main(args):
    #global_step=tf.train.get_or_create_global_step()
    #tower_grads = []V
    #X = tf.placeholder(tf.float32, [None, num_input])
    #Y = tf.placeholder(tf.float32, [None, num_classes])        
    network = inception_resnet_v2
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        train_set = get_dataset(args.data_dir)
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
        #FIFOQueue 先进现出队列
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                     dtypes=[tf.string, tf.int64],
                                     shapes=[(3,), (3,)],
                                     shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames,label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents,channels=3)
                if args.random_crop:
                    image = tf.random_crop(image,[args.image_size,args.image_size,3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image,args.image_size,args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
                image.set_shape((args.image_size,args.image_size,3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images,label])
        image_batch,labels_batch = tf.train.batch_join(
            images_and_labels,batch_size = batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        prelogits,_ = network.inference(image_batch,args.keep_probability,phase_train=phase_train_placeholder,
                bottleneck_layer_size=args.embedding_size,weight_decay=args.weight_decay)
        embeddings = tf.nn.l2_normalize(prelogits,1,1e-10,name = 'embeddings')
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
        triplet_loss = triplet_loss_f(anchor,positive,negative,args.alpha)
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
             args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')
        train_op = facenet_train(total_loss, global_step, args.optimizer,
                    learning_rate, args.moving_average_decay, tf.global_variables())
        saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=3)
        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord,sess=sess)

        with sess.as_default():
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step,feed_dict = None)
                epoch = step // args.epoch_size
                train(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
                      batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step,
                      embeddings, total_loss, train_op,
                      args.embedding_size, anchor, positive, negative, triplet_loss)

def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step,
          embeddings, loss, train_op, 
          embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0
    lr = args.learning_rate
    while batch_number < args.epoch_size:
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch,args.images_per_person)
        print("Running forward pass on sampled images:",end='')
        start_time = time.time()
        #90*40
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples),(-1,3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1),(-1,3))
        sess.run(enqueue_op,{image_paths_placeholder:image_paths_array,labels_placeholder:labels_array})
        #(90*40,90)
        emb_array = np.zeros((nrof_examples,embedding_size))
        #40
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab,:] = emb
        print('%.3f' % (time.time()-start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
            image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
            (nrof_random_negs, nrof_triplets, selection_time))
        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            emb_array[lab,:] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)
    return step        

def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j -1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings),1)
            for pair in xrange(1,nrof_images):
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0]
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    trip_idx += 1
                num_trips += 1
        emb_start_idx += nrof_images
    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9,name = 'arg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
        return loss_averages_op
def facenet_train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    loss_averages_op = _add_loss_summaries(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        if optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')
        grads = opt.compute_gradients(total_loss,update_gradient_vars)
    apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name,var)
    if log_histograms:
        for grad,var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients',grad)
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op,variable_averages_op]):
        train_op = tf.no_op(name = 'train')
    return train_op





def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,
            help='Path to the data directory containing aligned face patches.',
            default = '/data/vggface2/train')
    parser.add_argument('--data_list',type=str,
            help='Path to the list of face imgs.',
            default = '/data/vggface2/train_list.txt')
    parser.add_argument('--image_size', type=int,
            help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--random_crop',
            help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' + 'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip',
            help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--batch_size', type=int,
            help='Number of images to process in a batch.', default=90)
    parser.add_argument('--keep_probability', type=float,
            help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
            help='L2 weight regularization.', default=0.0)
    parser.add_argument('--embedding_size', type=int,
            help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--learning_rate', type=float,
            help='Initial learning rate. If set to a negative value a learning rate ' +
            'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
            help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
            help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
            help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--epoch_size', type=int,
            help='Number of batches per epoch.', default=1000)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
            help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--alpha', type=float,
            help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--max_nrof_epochs', type=int,
            help='Number of epochs to run.', default=500)
    parser.add_argument('--people_per_batch', type=int,
            help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
            help='Number of images per person.', default=40)
    return parser.parse_args(argv)

class ImageClass():
    #ImageClass:保存一个人脸的id和id对应的所有图片的绝对路径
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
    def __str__(self):
        return self.name + ',' + str(len(self.image_paths)) + 'images'
    def __len__(self):
        return len(self.image_paths)

def get_image_paths(facedir):
    #获得传入路径下所有图片的绝对路径
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_dataset(path,has_class_directories=True):
    #获取数据集所有人脸id和其对应图片的路径
    dataset = []
    path_exp= os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                if os.path.isdir(os.path.join(path_exp,path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
    return dataset

def sample_people(dataset,people_per_batch,images_per_person):
    #打乱顺序，选len（num_per_class）个id，对于第i个id，图片集合为image_paths[i]，图片数量为num_per_class[i]
    nrof_images = people_per_batch * images_per_person
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i=0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    while len(image_paths)<nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class,images_per_person,nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i+=1
    return image_paths,num_per_class
def triplet_loss_f(anchor,positive,negative,alpha):
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),1)
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss,0.0),0)
    return loss

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
