from flask import Flask, request, jsonify
from model_files.ml_model import predict_image
import numpy as np
import cv2

app = Flask('cat_dog_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    # receive json
    res = request.get_json()

    # conver json to dictionary
    image_array_dict = dict(res)

    # receive image array already in list format
    image_list = image_array_dict.get('image')

    print('\n\n*************************************')
    print('image list received is: ', image_list)
    print('\n\n*************************************')

    # # convert image from list to np array
    # image_ndarray = np.array(image_list)

    # convert image bytes 1D np array 
    nparr = np.fromstring(bytes(image_list), np.uint8)

    print('\n\n*************************************')
    print('nparr is: ', nparr)
    print('\n\n*************************************')

    # convert 1D np array to np 3D array
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print('\n\n*************************************')
    print('numpy 3D is: ', img_np)
    print('\n\n*************************************')

    # resize image using open cv and make prediction
    try:
        img_resize = cv2.resize(img_np, (300, 300), interpolation=cv2.INTER_AREA)
        preds = predict_image(img_resize)
        response = {
            'prediction': preds.tolist(),
        }
        return jsonify(response)

    except Exception as e:
        print("The caught error is: ",str(e))
        
        return jsonify({
            'message': 'An error occurred: '+str(e)
        })

    #return show_message('This message is returned grrr')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)