import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

df = pd.read_excel("initial_values\merge_excel.xlsx")
df = df.dropna(how="all")
df = df.dropna(axis=1,how="all")
df = df.drop(columns=['First Name', 'Last Name', 'User Name', 'Created On', 'Creator User ID', 'Date Of Birth', 'Gender', 'Last Modified On', 'Last Period Date', 'Socio-economic status', 'Patient Document Id','Patient ID','Organisation Patient ID'])
df = df.fillna(0)

print('\n')
print(df)

imgList = os.listdir("dataset")
imgListBase = []

for i in imgList:

    imgListBase.append(os.path.splitext(i)[0])

for col in df.columns:

    if df[col].dtype == np.float64:
        df[col] = df[col].astype(int)
    
X = df.drop(columns='Haemoglobin (gm/dL)')
y = df['Haemoglobin (gm/dL)']

# X = np.array(X)
y = to_categorical(y).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(24,))) # Input layer with 784 features
model.add(Dense(64, activation='relu')) # Hidden layer
model.add(Dense(24, activation='softmax'))

reduce_lr = ReduceLROnPlateau(monitor='categorical_accuracy',
                              factor=0.1,
    patience=15,
    verbose=0,
    mode='max',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0)
  
terminate_callback = EarlyStopping(monitor='categorical_accuracy', patience=50,mode='max')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, batch_size=256,callbacks=[terminate_callback,reduce_lr])
model.save('Intermediate Hemoglobin.h5')

loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)

model = load_model('Intermediate Hemoglobin.h5')
intermediateHgb = model.predict(X)

