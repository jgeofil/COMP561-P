
from cStringIO import StringIO
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data = []
SNPS = 100000

print 'Reading txt file...'

with open('data/in.txt') as f:
    lines = f.readlines()
    i = 0
    for l in lines:
        if l[0] != '#' and i < SNPS:
            i+=1
            s = l.split()
            data.append(s[9:])

print 'Opened'
LABELS = np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,
3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,
7,7,7,7,7,7,7,7])

data = np.array(data).transpose()


from keras.layers import Input, Dense
from keras.models import Model


# this is our input placeholder
inputData = Input(shape=(data.shape[1],))

# this is the size of our encoded representations
encoding_dim = 1500  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(inputData)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(data.shape[1], activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=inputData, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=inputData, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='sgd', loss='binary_crossentropy')

#train, val, train, val = train_test_split(data, data, test_size=1, random_state=42)
train = data


autoencoder.fit(train, train,
                nb_epoch=5,
                shuffle=False,
                batch_size=32,
                #validation_data=(val.T, val.T),
                verbose=1)

                # encode and decode some digits
# note that we take them from the *test* set
encoded_data = encoder.predict(data)

print encoded_data.shape

loo = LeaveOneOut()
res = []
for train, test in loo.split(encoded_data):
    clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, verbose=True)
    clf.fit(encoded_data[train], LABELS[train])
    res.append(clf.predict(encoded_data[test])[0])
    print res[test], LABELS[test]
for i in range(len(LABELS)):
    print res[i], LABELS[i]

print (np.array(res) == LABELS).sum()/float(len(data))
