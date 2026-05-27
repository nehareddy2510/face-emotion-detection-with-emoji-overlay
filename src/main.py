import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("model/emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model/emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Load emoji images (with or without alpha channel)
emoji_images = {
    0: cv2.imread('emoji/angry.png', -1),
    1: cv2.imread('emoji/disgust.png',-1),
    2: cv2.imread('emoji/fear.png', -1),
    3: cv2.imread('emoji/happy.png', -1),
    4: cv2.imread('emoji/neutral.png', -1),
    5: cv2.imread('emoji/sad.png', -1),
    6: cv2.imread('emoji/surprise.png', -1)
}

# Function to check if an image has an alpha channel
def has_alpha_channel(image):
    if image is None:
        return False
    return image.shape[2] == 4  # Check if there are 4 channels (RGBA)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            emoji = emoji_images[pred.argmax()]
            
            # Check if the emoji has an alpha channel
            if has_alpha_channel(emoji):
                # Split the emoji into its channels
                emoji_resized = cv2.resize(emoji, (r, s))
                alpha_channel = emoji_resized[:, :, 3] / 255.0
                emoji_resized = emoji_resized[:, :, 0:3]  # RGB channels
                
                x_offset = p
                y_offset = q
                
                for c in range(0, 3):
                    im[y_offset:y_offset + s, x_offset:x_offset + r, c] = \
                        im[y_offset:y_offset + s, x_offset:x_offset + r, c] * (1 - alpha_channel) + \
                        emoji_resized[:, :, c] * alpha_channel
            else:
                # If no alpha channel, use just the emoji RGB
                emoji_resized = cv2.resize(emoji, (r, s))
                x_offset = p
                y_offset = q
                for c in range(0, 3):
                    im[y_offset:y_offset + s, x_offset:x_offset + r, c] = \
                        im[y_offset:y_offset + s, x_offset:x_offset + r, c] * (1 - 0.5) + \
                        emoji_resized[:, :, c] * 0.5  # Blend with 50% opacity

            cv2.putText(im, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        cv2.imshow("Output", im)

        if cv2.waitKey(1) == ord('x'):
            break

    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()
