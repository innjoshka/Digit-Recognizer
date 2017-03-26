import pandas as pd
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


TRAIN_DIR = '/Users/ivanmac/Desktop/Kaggle/Digit Recognition (MNIST) dataset/train.csv'
TEST_DIR = '/Users/ivanmac/Desktop/Kaggle/Digit Recognition (MNIST) dataset/test.csv'

train_data = pd.read_csv(TRAIN_DIR)
test_x = pd.read_csv(TEST_DIR)

#Data preparations for tflearn
X = np.array(train_data.drop(['label'], 1 )) #FUCKING INDEX INPLACE=TRUE
Y = (train_data['label'])
test_x = np.array(test_x)

X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

Y = LabelEncoder().fit_transform(Y)[:, None]
Y = OneHotEncoder().fit_transform(Y).todense()


#creating computational graph
CNN = input_data(shape=[None, 28, 28, 1], name='input')

CNN = conv_2d(CNN, 32, 2, activation='relu')
CNN = max_pool_2d(CNN, 2)

CNN = conv_2d(CNN, 64, 2, activation='relu')
CNN = max_pool_2d(CNN, 2)

CNN = fully_connected(CNN, 1024, activation='relu')
CNN = dropout(CNN, 0.9)

CNN = fully_connected(CNN, 10, activation='softmax')
CNN = regression(CNN, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy', name='targets')


model = tflearn.DNN(CNN)
model.fit({'input': X}, {'targets': Y}, n_epoch=10, batch_size=96, snapshot_step=1000, show_metric=True)

model.save('CNN.model')

#model.load('CNN.model')


#making predictions
id=[]
predictions = []
predicts = model.predict(test_x)
for i, values in enumerate(predicts):
	predict_digit=values.index(max(values))
	id.append(i+1)
	predictions.append(predict_digit)


submission = pd.DataFrame({
	"ImageId": id,
	"Label": predictions
	})

submission.to_csv("digit_submission.csv", index=False)
