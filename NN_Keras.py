import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def normalise(data):
    data=[float(x) for x in data]
    
    data = [(x/max(data)) for x in data]

def reads_file(filename):
    df=pd.read_csv(filename, delimiter=',', header=None)
    
    return df

batch_size = 100
num_classes = 2
epochs = 20
input_features = 30


data=reads_file('data.csv')

labels=data[data.columns[1]]

labels=[ord(x) for x in labels]

labels=pd.get_dummies(labels)
#labels = labels.iloc[0,]
print(labels.head())

data=data.drop(data.columns[[0]], axis=1)
data=data.drop(data.columns[[0]], axis=1)

#data=normalise(data)

print(data.head())

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(data, labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)
score = model.evaluate(data, labels, verbose=0)
model.save('cancer.h5')
for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays

print(weights)
print('Test accuracy:', score[1])