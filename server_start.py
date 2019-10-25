# coding:utf-8
import os
import cv2
import dlib
import numpy as np
from flask import Flask, request
from keras.models import load_model
from imutils import face_utils
import time
import tool
import config

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model = load_model('eye.hdf5')


def merge_image(lack_face_img, face_img, rect):
    (x, y, w, h) = rect
    #x += 10
    #y += 10
    height, width, _ = face_img.shape
    for i in range(height):
        for j in range(width):
            lack_face_img[i + y, j + x, 0] = face_img[i, j, 0]
            lack_face_img[i + y, j + x, 1] = face_img[i, j, 1]
            lack_face_img[i + y, j + x, 2] = face_img[i, j, 2]
    return lack_face_img

def detect(img, cascade=face_cascade, minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)

    # if it doesn't return rectangle return array
    # with zero lenght
    if len(rects) == 0:
        return []

    #  convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]

    return rects


def cropEyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect the face at grayscale image
    te = detect(gray)

    # if the face detector doesn't detect face
    # return None, else if detects more than one faces
    # keep the bigger and if it is only one keep one dim
    if len(te) == 0:
        return None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te

    # keep the face region from the whole frame
    face_rect = dlib.rectangle(left=int(face[0]), top=int(face[1]),
                               right=int(face[2]), bottom=int(face[3]))
    # determine the facial landmarks for the face region
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)

    #  grab the indexes of the facial landmarks for the left and
    #  right eye, respectively
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # extract the left and right eye coordinates
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]


    # keep the upper and the lower limit of the eye
    # and compute the height
    l_uppery = min(leftEye[1:3, 1])
    l_lowy = max(leftEye[4:, 1])
    l_dify = abs(l_uppery - l_lowy)

    # compute the width of the eye
    lw = (leftEye[3][0] - leftEye[0][0])

    # we want the image for the cnn to be (26,34)
    # so we add the half of the difference at x and y
    # axis from the width at height respectively left-right
    # and up-down
    # minxl = (leftEye[0][0] - ((34 - lw) / 2))
    # maxxl = (leftEye[3][0] + ((34 - lw) / 2))
    # minyl = (l_uppery - ((26 - l_dify) / 2))
    # maxyl = (l_lowy + ((26 - l_dify) / 2))
    minxl = leftEye[0][0]
    maxxl = leftEye[3][0]
    higl = (maxxl - minxl) / (24 / 24)
    minyl = l_uppery + (((l_lowy - l_uppery) - higl) / 2)
    maxyl = minyl + higl

    # crop the eye rectangle from the frame
    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]

    # same as left eye at right eye
    r_uppery = min(rightEye[1:3, 1])
    r_lowy = max(rightEye[4:, 1])
    r_dify = abs(r_uppery - r_lowy)
    rw = (rightEye[3][0] - rightEye[0][0])
    # minxr = (rightEye[0][0] - ((34 - rw) / 2))
    # maxxr = (rightEye[3][0] + ((34 - rw) / 2))
    # minyr = (r_uppery - ((26 - r_dify) / 2))
    # maxyr = (r_lowy + ((26 - r_dify) / 2))
    minxr = rightEye[0][0]
    maxxr = rightEye[3][0]
    higr = (maxxr - minxr) / (24 / 24)
    minyr = r_uppery + (((r_lowy - r_uppery) - higr) / 2)
    maxyr = minyr + higr

    right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
    right_eye_rect = right_eye_rect.astype(int)
    right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

    # if it doesn't detect left or right eye return None
    if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
        return None
    # resize for the conv net
    left_eye_image = cv2.resize(left_eye_image, (24, 24))
    right_eye_image = cv2.resize(right_eye_image, (24, 24))
    right_eye_image = cv2.flip(right_eye_image, 1)
    # return left and right eye
    return left_eye_image, right_eye_image


# make the image to have the same format as at training
def cnnPreprocess(img):
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/api/upload', methods=['post'])
def upload():
    images = request.files.getlist('img[]')  # 获取上传的文件
    print("******", images)

    if len(images):
        im = []
        # save
        for i in range(len(images)):
            image = images[i]
            input = config.upload_dir + image.filename

            im.append({
                'raw': image,
                'path': input,
            })

            try:
                image.save(input)
                #tool.img_resize_to_target_white(input, output)
                #return send_from_directory('', output, as_attachment=True)
            except Exception as e:
                print("Error: ", e)
            # finally:
            #     if os.path.exists(input):
            #         os.remove(input)
            #     if os.path.exists(output):
            #         os.remove(output)

        # read first image as base image
        first_image_path = im[0]['path']
        first_image = cv2.imread(first_image_path)

        gray = cv2.cvtColor(first_image, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray)

        # save first image faces rect
        im[0]['faces'] = faces

        mini_face = []
        for i in range(len(faces)):
            mini_face.append({
                'ok': {},
                'face_list': []
            })

        for j in range(len(im)):
            img = cv2.imread(im[j]['path'])
            for i in range(len(faces)):
                face = faces[i]
                rect = np.rint([face[0], face[1], face[0] + face[2], face[1] + face[3]])
                rect = rect.astype(int)
                f = img[rect[1]:rect[3], rect[0]:rect[2]]
                mini_face[i]['face_list'].append({
                    'image': f,
                    'rect': face,
                    'np_rect': rect
                })

        for i in range(len(mini_face)):
            faces = mini_face[i]['face_list']
            for j in range(len(faces)):
                face = faces[j]
                max_face_image = tool.img_resize_to_target_white(face['image'])
                left_eye = right_eye = 0
                prediction = 0

                # detect eyes
                eyes = cropEyes(max_face_image)
                if eyes is None:
                    print("eyes is null")
                else:
                    left_eye, right_eye = eyes
                    prediction = (model.predict(cnnPreprocess(left_eye)) + model.predict(
                        cnnPreprocess(right_eye))) / 2.0

                mini_face[i]['face_list'][j]['prediction'] = prediction

                if 'prediction' not in mini_face[i]['ok']:
                    mini_face[i]['ok']['face'] = face
                    mini_face[i]['ok']['prediction'] = prediction
                else:
                    if mini_face[i]['ok']['prediction'] < prediction:
                        mini_face[i]['ok']['face'] = face
                        mini_face[i]['ok']['prediction'] = prediction

        # merge
        for item in mini_face:
            ok = item['ok']
            print(ok['prediction'])
            first_image = merge_image(first_image, ok['face']['image'], ok['face']['rect'])

        cv2.putText(first_image, 'mix@magvii ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.imwrite("mix/dist/img/merged_image.jpg", first_image)
        return '{"code": 1000}'
    else:
        return '{"code": 9999}'


def mk_img_dir():
    if not os.path.exists(config.upload_dir):
        os.mkdir(config.upload_dir)
    if not os.path.exists(config.static_dir):
        os.mkdir(config.static_dir)
    pass


if __name__ == '__main__':
    mk_img_dir()
    app.run(port=config.port)
    pass
