from keras.models import Sequential
from keras.layers import Input, Convolution2D, Lambda, merge, Dense, Flatten,MaxPooling2D,BatchNormalization,Dropout
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam,RMSprop
import numpy as np
import pickle
import random
from sklearn.preprocessing import scale
with open('/home/xm/桌面/img/data_1/small sample/sample.txt','rb') as f:
    data = pickle.load(f)

X_train,y_train,X_test,y_test = data['X_train'],data['y_train'],data['X_test'],data['y_test']
##(X_train, y_train), (X_test, y_test) = mnist.load_data()
##X_train = X_train.reshape(60000, 784) #将图片向量化
##X_test = X_test.reshape(10000, 784)
##X_train = X_train.astype('float32')
##X_test = X_test.astype('float32')
##X_train /= 255 # 归一化
##X_test /= 255
print('data loaded.....')

def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    model = Sequential()
    model.add(BatchNormalization(input_shape = input_dim))
    model.add(Convolution2D(32,5,5,activation='relu',border_mode = 'valid',W_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(64,3,3,border_mode = 'valid',activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(128,1,1,border_mode = 'valid',activation='relu'))
    model.add(Flatten())
    model.add(Dense(1000,activation="relu",))
    model.add(Dense(396,activation = 'relu'))
    return model
input_dim = (28,28,3)
base_network = create_base_network(input_dim)

input_a = Input(shape=input_dim)
input_b = Input(shape=input_dim)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

##distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

digit_indices = [np.where(y_train == i)[0] for i in range(396)]
print('length of digit_indices:',digit_indices[4])

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = [] #一会儿一对对的样本要放在这里
    labels = []
    n = min([len(digit_indices[d]) for d in range(396)]) - 1
    print('n:',n)
    for d in range(396):
        #对第d类抽取正负样本
        for i in range(n):
            # 遍历d类的样本，取临近的两个样本为正样本对
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            # randrange会产生1~9之间的随机数，含1和9
            inc = random.randrange(0, 396)
            # (d+inc)%10一定不是d，用来保证负样本对的图片绝不会来自同一个类
            dn = (d + inc) % 396
            # 在d类和dn类中分别取i样本构成负样本对
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            # 添加正负样本标签
            labels += [1, 0]
    return np.array(pairs), np.array(labels)
#训练集的pair
digit_indices = [np.where(y_train == i)[0] for i in range(396)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices)
print('train data prepared........')
print('tr_pairs.shape:',tr_pairs.shape)
#测试集的pair
digit_indices = [np.where(y_test == i)[0] for i in range(396)]
te_pairs, te_y = create_pairs(X_test, digit_indices)
print('test data prepared........')
print('te_pairs shape:',te_pairs.shape)

nb_epoch = 300
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=32,
          epochs=nb_epoch)

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()
# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
print(pred.shape)
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
