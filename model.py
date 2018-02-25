import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
import os

# supress warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
'''
Training data (name, min, max)
Market 1 4, one hot
Day 1 729, maybe one hot
Stock 0 3022, maybe one hot
x0 0.0 4998275.0379
x1 1.29522219386e-05 2946.26389986
x2 0.00031347643418 677.912472849
x3A 0.0 0.0283150876226
x3B 0.0 0.0651416193937
x3C 0.0 0.109440455853
x3D 0.0 0.206560349872
x3E 0.0 0.361169253845
x4 4.7396e-06 0.1022
x5 0.0 0.0358925753318
x6 1.0 734914.404967
y -0.0710989470203 0.0666166666679
Weight 0.00279733959939 694.001930036
'''
mins = [1.0, 0.0, 0.0, 1.29522219386e-05, 0.00031347643418, 0.0, 0.0, 0.0, 0.0, 0.0, 4.7396e-06, 0.0, 1.0, -0.0710989470203, 0.00279733959939]
maxs = [729.0, 3022.0, 4998275.0379, 2946.26389986, 677.912472849, 0.0283150876226, 0.0651416193937, 0.109440455853, 0.206560349872, 0.361169253845, 0.1022, 0.0358925753318, 734914.404967, 0.0666166666679, 694.001930036]

def get_training_data(frame):
    for t in frame.itertuples():
        t = list(t)
        x = t[1:15]
        y = t[15:]
        yield x, y


def prepare_features(x):
    new_xs = []
    for vec in x:
        new_x = [0, 0, 0, 0]

        # one hot encode market
        new_x[vec[0]-1] = 1

        # put others in range -1 to 1
        for i in xrange(1, len(vec)):
            mid = (mins[i-1]+maxs[i-1])/2.
            new_x.append((vec[i]-mid)/(maxs[i-1]-mid))
        new_xs.append(new_x)

    return np.array(new_xs)

def prepare_labels(y):
    labels, weights = [], []
    for vec in y:
        # mid_label = (mins[-2]+maxs[-2])/2.
        # labels.append([(vec[0]-mid_label)/(maxs[-2]-mid_label)])
        labels.append([vec[0]])
        weights.append([vec[1]])
    return np.array(labels), np.array(weights)



def prepare_vectors(x, y):
    x = prepare_features(x)
    l, w = prepare_labels(y)
    return x, l, w

def batch_training_data(frame):
    batch_x = []
    batch_y = []
    for x, y in get_training_data(frame):
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) == BATCH_SIZE:
            yield prepare_vectors(batch_x, batch_y)
            batch_x = []
            batch_y = []


FEATURE_LENGTH = 13 + 4 # 12 features + one hot encode market
BATCH_SIZE = 32

class GangstaModel(object):

    def __init__(self, chkpt=None):
        # load from chkpt if exists
        self.create_graph()
        if chkpt:
            self.load(chkpt)

    def create_graph(self):
        self.session = tf.Session()

        self.input_layer = tf.placeholder(tf.float32, [None, FEATURE_LENGTH])
        self.labels = tf.placeholder(tf.float32, [None, 1])
        self.label_weights = tf.placeholder(tf.float32, [None, 1])

        layer1 = tf.layers.dense(self.input_layer, 300, activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, 300, activation=tf.nn.relu)
        layer3 = tf.layers.dense(layer2, 300, activation=tf.nn.relu)
        layer4 = tf.layers.dense(layer3, 300, activation=tf.nn.relu)
        layer5 = tf.layers.dense(layer4, 300, activation=tf.nn.relu)
        self.out_layer = tf.layers.dense(layer5, 1)

        # loss
        # for loss need to unreg
        # mid_label = (mins[-2]+maxs[-2])/2.
        # self.out_unreg = self.out_layer*(maxs[-2]-mid_label) + mid_label

        self.loss = tf.reduce_sum(self.label_weights * tf.squared_difference(self.labels, self.out_layer))
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.min_loss = self.optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        self.session.run(initializer)

        self.saver = tf.train.Saver()

    def predict(self, features):
        return self.session.run([self.out_layer], feed_dict={
            self.input_layer: features
        })

    def train_batch(self, features, labels, label_weights):
        loss, _ = self.session.run([self.loss, self.min_loss], feed_dict={
            self.input_layer: features,
            self.labels: labels,
            self.label_weights: label_weights
        })

    def batch_loss(self, features, labels, label_weights):
        loss = self.session.run([self.loss], feed_dict={
            self.input_layer: features,
            self.labels: labels,
            self.label_weights: label_weights
        })
        return loss


    def train_epoch(self, frame):
        for x, labels, weights in tqdm.tqdm(batch_training_data(frame), total=int(frame.shape[0]/BATCH_SIZE)):
            assert not np.any(np.isnan(x))
            assert not np.any(np.isnan(labels))
            assert not np.any(np.isnan(weights))
            self.train_batch(x, labels, weights)

    def load(self, name):
        self.saver.restore(self.session, name)

    def save(self, name='model/gangstamodel'):
        self.saver.save(self.session, name)
        print 'Saved model to {}!'.format(name)


def see_total_loss(df, model):
    batch_x = []
    batch_y = []
    for x, y in get_training_data(df):
        batch_x.append(x)
        batch_y.append(y)
    loss = model.batch_loss(*prepare_vectors(batch_x, batch_y))
    return loss 


def trainer():
    df = pd.read_csv('train.csv', index_col=0)
    df = df.fillna(0)
    model = GangstaModel(chkpt=None)

    num_epochs = 100
    best_loss = 10000
    for i in xrange(num_epochs):
        print 'EPOCH', i+1
        # trains on scrambled df
        model.train_epoch(df.sample(frac=1))
        loss = see_total_loss(df, model)[0]
        print "Loss:", loss
        if loss < best_loss:
            model.save('model1/gangstamodel{}'.format(loss))
            best_loss = loss


def create_test_csv(test_filename='test.csv'):
    print 'Loading test file...'
    df = pd.read_csv(test_filename, index_col=0).fillna(0)
    model = GangstaModel(chkpt='model1/gangstamodel0.998178064823')

    # read data
    vals = []
    batch = []
    print 'Loading batch list...'
    for t in tqdm.tqdm(df.itertuples(), total=df.shape[0]):
        batch.append(list(t)[1:])
    print 'Preparing features for predicting...'
    x = prepare_features(batch)
    print 'Predicting...'
    prediction = model.predict(x)[0]
    y = [val[0] for val in prediction]
    print 'Saving...'
    df = pd.DataFrame(y, columns=['y'])
    df.index.name = 'Index'
    df.to_csv('answers.csv', header=True)
    print 'Saved to answers.csv'


if __name__ == '__main__':
    # create_test_csv()
    trainer()
