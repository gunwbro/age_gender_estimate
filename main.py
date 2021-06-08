import os.path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from PIL import Image
import sys
from sklearn.model_selection import train_test_split

data0 = pd.read_csv('input/fold_0_data.txt', sep="\t")
data1 = pd.read_csv('input/fold_1_data.txt', sep="\t")
data2 = pd.read_csv('input/fold_2_data.txt', sep="\t")
data3 = pd.read_csv('input/fold_3_data.txt', sep="\t")
data4 = pd.read_csv('input/fold_4_data.txt', sep="\t")

total_data = pd.concat([data1, data0, data2, data3, data4], ignore_index=True)
df = total_data[['age', 'gender', 'x', 'y', 'dx', 'dy']].copy()
path = []
for row in total_data.iterrows():
    img = "input/faces/" + row[1].user_id + "/coarse_tilt_aligned_face." + str(row[1].face_id) + "." + row[
        1].original_image
    path.append(img)

df['path'] = path

age_mapping = [('(0, 2)', '0-2'), ('2', '0-2'), ('3', '0-2'), ('(4, 6)', '4-6'), ('(8, 12)', '8-13'), ('13', '8-13'),
               ('22', '15-20'), ('(8, 23)', '15-20'), ('23', '25-32'), ('(15, 20)', '15-20'), ('(25, 32)', '25-32'),
               ('(27, 32)', '25-32'), ('32', '25-32'), ('34', '25-32'), ('29', '25-32'), ('(38, 42)', '38-43'),
               ('35', '38-43'), ('36', '38-43'), ('42', '48-53'), ('45', '38-43'), ('(38, 43)', '38-43'),
               ('(38, 42)', '38-43'), ('(38, 48)', '48-53'), ('46', '48-53'), ('(48, 53)', '48-53'), ('55', '48-53'),
               ('56', '48-53'), ('(60, 100)', '60+'), ('57', '60+'), ('58', '60+')]

age_mapping_dict = {each[0]: each[1] for each in age_mapping}
drop_labels = []
for idx, each in enumerate(df.age):
    if each == 'None':
        drop_labels.append(idx)
    else:
        df.age.loc[idx] = age_mapping_dict[each]

df = df.drop(labels=drop_labels, axis=0)
df = df.dropna()
unbiased_data = df[df.gender != 'u'].copy()

gender_to_label_map = {
    'f': 0,
    'm': 1
}

age_to_label_map = {
    '0-2': 0,
    '4-6': 1,
    '8-13': 2,
    '15-20': 3,
    '25-32': 4,
    '38-43': 5,
    '48-53': 6,
    '60+': 7
}

unbiased_data['age'] = unbiased_data['age'].apply(lambda age: age_to_label_map[age])
unbiased_data['gender'] = unbiased_data['gender'].apply(lambda g: gender_to_label_map[g])

X = unbiased_data[['path']]
y_gender = unbiased_data[['gender']]
y_age = unbiased_data[['age']]
X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.3,
                                                                                random_state=42)
X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.3, random_state=42)

gender_train_images = []
gender_test_images = []

age_train_images = []
age_test_images = []

for row in X_age_train.iterrows():
    if os.path.isfile(row[1].path):
        image = Image.open(row[1].path)
        image = image.resize((227, 227))
        data = np.asarray(image)
        age_train_images.append(data)

for row in X_age_test.iterrows():
    if os.path.isfile(row[1].path):
        image = Image.open(row[1].path)
        image = image.resize((227, 227))
        data = np.asarray(image)
        age_test_images.append(data)

for row in X_gender_train.iterrows():
    if os.path.isfile(row[1].path):
        image = Image.open(row[1].path)
        image = image.resize((227, 227))
        data = np.asarray(image)
        gender_train_images.append(data)

for row in X_gender_test.iterrows():
    if os.path.isfile(row[1].path):
        image = Image.open(row[1].path)
        image = image.resize((227, 227))
        data = np.asarray(image)
        gender_test_images.append(data)

gender_train_images = np.asarray(gender_train_images)
gender_test_images = np.asarray(gender_test_images)
age_train_images = np.asarray(age_train_images)
age_test_images = np.asarray(age_test_images)

genderModel = tf.keras.Sequential([
    L.Conv2D(96, (7, 7), activation='relu', input_shape=(227, 227, 3), strides=4, padding='valid'),
    L.MaxPooling2D((3, 3), strides=(2, 2)),
    # L.LayerNormalization(),
    L.Conv2D(256, (5, 5), activation='relu', strides=1, padding='same'),
    L.MaxPooling2D((3, 3), strides=(2, 2)),
    # L.LayerNormalization(),
    L.Conv2D(384, (3, 3), activation='relu', strides=1, padding='same'),
    L.MaxPooling2D((3, 3), strides=(2, 2)),
    L.Flatten(),
    L.Dense(512, activation='relu'),
    # L.Dropout(rate=0.5),
    L.Dense(512, activation='relu'),
    # L.Dropout(rate=0.5),
    L.Dense(1, activation='sigmoid')
])

ageModel = tf.keras.Sequential([
    L.Conv2D(96, (7, 7), activation='relu', input_shape=(227, 227, 3), strides=4, padding='valid'),
    L.MaxPooling2D((3, 3), strides=(2, 2)),
    # L.LayerNormalization(),
    L.Conv2D(256, (5, 5), activation='relu', strides=1, padding='same'),
    L.MaxPooling2D((3, 3), strides=(2, 2)),
    # L.LayerNormalization(),
    L.Conv2D(384, (3, 3), activation='relu', strides=1, padding='same'),
    L.MaxPooling2D((3, 3), strides=(2, 2)),
    L.Flatten(),
    L.Dense(512, activation='relu'),
    # L.Dropout(rate=0.5),
    L.Dense(512, activation='relu'),
    # L.Dropout(rate=0.5),
    L.Dense(8, activation='sigmoid')
])

genderModel.summary()
ageModel.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

genderModel.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy'])

ageModel.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

genderHistory = genderModel.fit(
    gender_train_images, y_gender_train, epochs=15, validation_data=(gender_test_images, y_gender_test), batch_size=32,
    callbacks=[callback]
)

ageHistory = ageModel.fit(
    age_train_images, y_age_train, epochs=15, validation_data=(age_test_images, y_age_test), batch_size=32,
    callbacks=[callback]
)

loss, acc = genderModel.evaluate(gender_test_images, y_gender_test, verbose=2)
print('Gender Test loss: {}'.format(loss))
print('Gender Test Accuracy: {}'.format(acc))

loss, acc = ageModel.evaluate(age_test_images, y_age_test, verbose=2)
print('Age Test loss: {}'.format(loss))
print('Age Test Accuracy: {}'.format(acc))

if len(sys.argv) != 2:
    print("Insufficient arguments")
    sys.exit()
file_path = sys.argv[1]
print("File path : " + file_path)
image = Image.open(file_path)
image = image.resize((227, 227))
data = np.asarray(image)
test = [data]
test = np.asarray(test)

genderResult = genderModel.predict(test)
ageResult = ageModel.predict(test)

print(genderResult)
print(ageResult)
print('----------------------------')
if genderResult[0][0] >= 0.5:
    print('성별 : 남성')
else:
    print('성별 : 여성')

print('나이 : ', [k for k, v in age_to_label_map.items() if v == np.argmax(ageResult[0])][0])
