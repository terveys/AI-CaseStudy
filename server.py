from flask import Flask, request, Response
import os

app = Flask(__name__)
model = None
labels = [ 'No Pneumonia', 'Bacterial Pneumonia', 'Viral Pneumonia' ]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return 'POST Pneumonia X-Ray image with name "image"'
    else:
        imagefile = request.files.get('image')
        if imagefile is None:
            return Response(None, 400)
        if not os.path.exists('uploads'):
            os.mkdir('uploads')
        filepath = os.path.join('uploads', imagefile.filename)
        imagefile.save(filepath)
        print(model)
        from PIL import Image
        import numpy as np
        image = Image.open(filepath)
        if image is None:
            raise Exception('Invalid image input')

        image = image.resize((180, 320))
        image = image.convert('RGB')
        image = np.array(image)
        image = image / 255
        input = np.array([ image ])
        print(input.shape)
        output = model.predict(input)
        result = output[0]
        print (result)
        labelindex = np.argmax(result)
        classification = labels[labelindex]
        return Response(classification, 200)

if __name__ == '__main__':
    import tensorflow as tf
    model = tf.keras.models.load_model('trained_model')
    model = tf.keras.models.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    app.run()