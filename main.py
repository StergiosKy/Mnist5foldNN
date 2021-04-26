import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.regularizers import l2

# Set parameters
epochs = 100
# values are 10, 397 or 794
hidden_layer_neurons = 397
learning_rate = 0.05
weight_momentum = 0.6
weight_decay = 0.0
r = 0.9
batchsize = 512
# this flag is used to determine if we do normalization or Standardization
# we chose to avoid centering
# False -> Standardization, True -> Normalization
normalization = False
# enable 2nd hidden layer
hidden_layer_2 = True
if hidden_layer_2:
    # we will try 3 numbers 184, 794, 792
    hidden_layer_2_neurons = 397
# CE vs MSE flag
# False -> MSE, True -> CE
use_CE = True

# Read dataset
dataset = np.loadtxt("mnist_train.csv", delimiter=",", skiprows=1)
# Split into input and output
output = dataset[:, 0]
# onehot encode the output to match the 10 output neurons
Y = OneHotEncoder(sparse=False).fit_transform(X=output.reshape(len(output), 1))
# remove the output from the input dataset
dataset = dataset[:, 1:]
# this below is redundant for the specific case
# test_dataset = np.loadtxt("mnist_test.csv", delimiter=",", skiprows=1)
print("Successfully read the dataset")


# Centering is removing the mean value of the dataset and subtracting it from every data point
# Warning the code below is really slow, exists just for demonstration purposes and is not used:
'''
X = np.zeros(shape=dataset.shape)
dataset_mean = np.mean(dataset)
for i in range(len(dataset)):
    X[i, :] = dataset[i, :] - dataset_mean
'''
# Normalization is the rescaling of the dataset from 0-255 to 0-1
if normalization:
    X = MinMaxScaler().fit_transform(X=dataset)
# Standardization, transform the dataset close to a Gauss distribution
# we use the default method yeo-johnson because 0 is included in the values, else we would've used box-cox
else:
    X = PowerTransformer().fit_transform(X=dataset)

print("Data pre-processing finished")

# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)
score_list = []
history_list = []
accuracy_list = []
error_list = []
history_loss = []
history_acc = []

for i, (train, test) in enumerate(kfold.split(X)):
    # Create model
    # Input is 784 length
    # output layer is of length 10 (numbers 0->9)
    # hidden layer 1 units for the example study are 10, 397, 794
    # hidden layer 2 is optimal_hidden_2 , it's double and it's half
    if hidden_layer_2:
        model = keras.Sequential([
            # hidden layer
            keras.layers.Dense(units=hidden_layer_neurons, activation='relu', input_shape=[28 * 28, ], kernel_regularizer=l2(r), bias_regularizer=l2(r)),
            # hidden layer 2
            keras.layers.Dense(units=hidden_layer_2_neurons, activation='relu', kernel_regularizer=l2(r), bias_regularizer=l2(r)),
            # output layer
            keras.layers.Dense(units=10, activation='softmax')
        ])
    else:
        model = keras.Sequential([
            # hidden layer
            keras.layers.Dense(units=hidden_layer_neurons, activation='relu', input_shape=[28 * 28, ]),
            # output layer
            keras.layers.Dense(units=10, activation='softmax')
        ])

    # redundant
    # early_stopping = keras.callbacks.EarlyStopping(monitor='rmse', min_delta=0, patience=0, verbose=1, mode='auto')
    keras.optimizers.SGD(lr=learning_rate, momentum=weight_momentum, decay=weight_decay, nesterov=False)

    if use_CE:
        # CE
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    else:
        # MSE
        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

    # Fit model
    history = model.fit(X[train], Y[train], epochs=epochs, batch_size=batchsize, verbose=0)
    history_list.append(history)
    history_loss.append(history.history['loss'])
    history_acc.append(history.history['accuracy'])
    # Evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=1)
    score_list.append(scores)
    print(f"Fold: {i+1}, accuracy: {100 * scores[1]}, loss: {scores[0]}")

    accuracy_list.append(scores[1] * 100)
    error_list.append(scores[0])

plt.figure(1)
# we use 5 fold
for i in range(5):
    # plot the data for the loss
    plt.plot(history_list[i].history['loss'], label=f'fold {i + 1}')

if hidden_layer_2:
    if use_CE:
        plt.title(f'Validation, Metric: CE, Batch size: {batchsize}, Hidden Layer 2 Neurons: {hidden_layer_2_neurons}')
    else:
        plt.title(f'Validation, Metric: MSE, Batch size: {batchsize}, Hidden Layer 2 Neurons: {hidden_layer_2_neurons}')
else:
    if use_CE:
        plt.title(f'Validation, Metric: CE, Batch size: {batchsize}, Hidden Layer Neurons: {hidden_layer_neurons}')
    else:
        plt.title(f'Validation, Metric: MSE, Batch size: {batchsize}, Hidden Layer Neurons: {hidden_layer_neurons}')
plt.xlabel('Epoch')
plt.ylabel('LOSS')
plt.legend()

plt.figure(2)
for i in range(5):
    # plot the data for the accuracy
    plt.plot(history_list[i].history['accuracy'], label=f'fold {i + 1}')


# show the plots
if hidden_layer_2:
    if use_CE:
        plt.title(f'Validation, Metric: CE, Batch size: {batchsize}, Hidden Layer 2 Neurons: {hidden_layer_2_neurons}')
    else:
        plt.title(f'Validation, Metric: MSE, Batch size: {batchsize}, Hidden Layer 2 Neurons: {hidden_layer_2_neurons}')
else:
    if use_CE:
        plt.title(f'Validation, Metric: CE, Batch size: {batchsize}, Hidden Layer Neurons: {hidden_layer_neurons}')
    else:
        plt.title(f'Validation, Metric: MSE, Batch size: {batchsize}, Hidden Layer Neurons: {hidden_layer_neurons}')
plt.xlabel('Epoch')
plt.ylabel('ACCURACY')
plt.legend()

print(f"Validation")
if use_CE:
    print("CE: ", np.mean(error_list))
else:
    print("MSE: ", np.mean(error_list))
print("Accuracy: ", np.mean(accuracy_list))

plt.figure(3)
average_loss = [sum(col)/len(col) for col in zip(*history_loss)]
average_acc = [sum(col)/len(col) for col in zip(*history_acc)]

# plot the data for the accuracy
plt.plot(average_acc)
if hidden_layer_2:
    if use_CE:
        plt.title(f'Validation, Metric: CE, Batch size: {batchsize}, Hidden Layer 2 Neurons: {hidden_layer_2_neurons}')
    else:
        plt.title(f'Validation, Metric: MSE, Batch size: {batchsize}, Hidden Layer 2 Neurons: {hidden_layer_2_neurons}')
else:
    if use_CE:
        plt.title(f'Validation, Metric: CE, Batch size: {batchsize}, Hidden Layer Neurons: {hidden_layer_neurons}')
    else:
        plt.title(f'Validation, Metric: MSE, Batch size: {batchsize}, Hidden Layer Neurons: {hidden_layer_neurons}')
plt.xlabel('Epoch')
plt.ylabel('ACCURACY')

plt.figure(4)
# plot the data for the accuracy
plt.plot(average_loss)
if hidden_layer_2:
    if use_CE:
        plt.title(f'Validation, Metric: CE, Batch size: {batchsize}, Hidden Layer 2 Neurons: {hidden_layer_2_neurons}')
    else:
        plt.title(f'Validation, Metric: MSE, Batch size: {batchsize}, Hidden Layer 2 Neurons: {hidden_layer_2_neurons}')
else:
    if use_CE:
        plt.title(f'Validation, Metric: CE, Batch size: {batchsize}, Hidden Layer Neurons: {hidden_layer_neurons}')
    else:
        plt.title(f'Validation, Metric: MSE, Batch size: {batchsize}, Hidden Layer Neurons: {hidden_layer_neurons}')
plt.xlabel('Epoch')
plt.ylabel('LOSS')

plt.show()
