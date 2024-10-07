import sys

# passの設定 (pip showで出てきた、LocationのPASSを以下に設定)
sys.path.append('/opt/anaconda3/envs/ZemiPy39/lib/python3.9/site-packages')

from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import os

# Flaskアプリケーションの作成
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# 学習済みのモデルを読み込みます。
model = load_model('my_model_v2_2.keras')

# 予測結果に対応するカテゴリ名とカロリーをリストで定義します。
class_names = ["セミ", "コオロギ", "バッタ"]
class_calories = [71.1, 63.9, 67.5]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 画像を読み込み、サイズを128x128に変更し、RGBに変換します。
        img = image.load_img(filepath, target_size=(128, 128)).convert("RGB")

        # 画像を数値の配列に変換し、配列を4次元テンソルに変換します（バッチサイズの次元を追加します）。
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x.astype('float32') / 255.0

        # モデルを使って予測を行います。
        predictions = model.predict(x)

        # 最も確率が高いカテゴリを選びます。
        predicted_class = np.argmax(predictions[0])

        # 予測結果とカロリーを設定します。
        name = f'\nこの虫は多分、{class_names[predicted_class]}だね！'
        protein = f'100gあたりタンパク質 {class_calories[predicted_class]} gもあるよ！'
        if predicted_class == 0:
            comment = f'幼虫はエビに似た味がして美味しいけど、 成虫は木の味しかしなくてまずいよ！'
        elif predicted_class == 1:
            comment = f'香ばしさのあるエビのような風味が特徴で、ナッツの味に似ているよ！'
        else:
            comment = f'複雑な味だからしっかり味付けをした方が 美味しいよ！'
        # 画像をbase64でエンコードします。
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return render_template('result.html', img_data=img_str, name=name , protein=protein , comment=comment)

    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
