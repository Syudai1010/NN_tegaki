import streamlit as st

import numpy as np

from PIL import Image, ImageOps

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.datasets import mnist

import os

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# タイトル表示

st.title("手書き数字認識アプリ")

 

# モデル保存ファイル名

model_path = "mnist_model.h5"

 

# モデルの作成またはロード

def create_and_train_model():

    # MNISTデータセットをロード

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

 

    # データの前処理

    train_images = train_images.astype('float32') / 255.0

    test_images = test_images.astype('float32') / 255.0

 

    # 画像の形状を28x28x1に変更（CNN用）

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))

    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

 

    # ラベルをone-hotエンコーディング

    train_labels = tf.keras.utils.to_categorical(train_labels, 10)

    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

 

    # モデルの構築（CNN）

    model = models.Sequential()

   

    # 畳み込み層

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

   

    # 全結合層

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))

 

    # モデルのコンパイル

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

 

    # モデルの学習

    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

 

    # テストデータでモデルを評価

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    st.write(f"テスト精度: {test_acc}")

 

    # モデルの保存

    model.save(model_path)

    st.success("モデルを保存しました。")

 

# モデルをロード

def load_model():

    if os.path.exists(model_path):

        model = tf.keras.models.load_model(model_path)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        st.success("保存されたモデルをロードし、再コンパイルしました。")

    else:

        st.error("モデルが見つかりません。トレーニングを行ってください。")

        model = None

    return model

 

 

# 画像を28x28に変換してモデルで予測する

def predict_digit(model, img):

    # 画像のリサイズとグレースケール変換

    img = img.resize((28, 28))  # リサイズ

    img = ImageOps.grayscale(img)  # グレースケール化

   

    # 背景が白、数字が黒であることを確認し、必要に応じて反転

    img = ImageOps.invert(img)

 

    img_array = np.array(img).astype('float32') / 255.0  # 正規化

    img_array = img_array.reshape(1, 28, 28, 1)  # CNN用に形状を変更

   

    # 画像を表示する（前処理後の画像）

    st.write("前処理された画像:")

    fig, ax = plt.subplots()

    ax.imshow(img_array.squeeze(), cmap='gray')

    ax.axis('off')

    st.pyplot(fig)

 

    # 予測

    prediction = model.predict(img_array)

    return np.argmax(prediction)

 

 

# トレーニングボタン

if st.button("モデルをトレーニング"):

    create_and_train_model()

 

# モデルのロード

model = load_model()

 

# 画像をアップロード

uploaded_file = st.file_uploader("手書き数字の画像をアップロードしてください", type=["png", "jpg", "jpeg"])

 

# アップロードされたファイルがあれば画像を表示し、予測結果を更新する

if uploaded_file is not None and model is not None:

    # 画像を表示

    image = Image.open(uploaded_file)

    st.image(image, caption="アップロードされた画像", use_column_width=True)

   

    # 予測

    prediction = predict_digit(model, image)

   

    # 予測結果を表示

    st.write(f"この画像は数字 {prediction} です。")
