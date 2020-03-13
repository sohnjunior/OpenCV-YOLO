import cv2
import os
import argparse
import numpy as np


def load_model(weight_path, config_path):
    """
    YOLO 모델을 불러온다.

    :param weight_path: path for weight file
    :param config_path: path for configuration file
    :return: YOLO model(net, output layer)
    """
    net = cv2.dnn.readNet(weight_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers


def label_color_setting(label_text_path):
    """
    분류할 클래스별로 색상을 정한다.

    :param label_text_path: 분류 가능한 클래스 목록을 담고있는 파일 경로
    :return: 클래스와 색상들
    """
    with open(label_text_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return classes, colors


def preprocess_input_image(image_path):
    """
    입력 이미지 전처리 과정을 거진다.

    :param image_path: 입력 이미지 경로
    :return: 전처리된 이미지 데이터와 기존의 이미지 데이
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    blob_img = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    return blob_img, img


def forward(net, output_layers, input_image):
    """
    feed forward

    :param net: YOLO net
    :param output_layers: output layers for YOLO
    :param input_image: input image
    :return: detection result
    """
    net.setInput(input_image)
    outs = net.forward(output_layers)
    return outs


def display_result(outs, input_image):
    """
    입력된 이미지에 인식된 객체들을 표시한다.

    :param outs: 인식된 객체들
    :param input_image: 입력 이미지
    :return: None
    """
    class_indices = []
    confidences = []
    boxes = []
    (H, W) = input_image.shape[:2]
    classes, colors = label_color_setting("yolov3.txt")
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_idx = np.argmax(scores)
            confidence = scores[class_idx]
            if confidence > 0.5:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_indices.append(class_idx)

    # 한 객체에 대해 중복되는 박스들을 지운다
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            (x, y, w, h) = boxes[i]
            label = str(classes[class_indices[i]])
            color = colors[i]
            cv2.rectangle(input_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(input_image, label, (x, y + 30), font, 2, color, 2)

    cv2.imshow("Image", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detection_image(input_image):
    """
    이미지를 대상으로 객체 검출을 수행한다.

    :param input_image: 입력 이미지
    :return: None
    """
    # check image path
    if not os.path.exists(input_image):
        print('[error] File Not Exist! check your file path')
        return

    (net, output_layers) = load_model(weight_path="yolov3.weights", config_path="yolov3.cfg")
    blob_img, original_img = preprocess_input_image(input_image)
    output = forward(net=net, output_layers=output_layers, input_image=blob_img)
    display_result(output, input_image=original_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Opencv-Yolov3 Object Detection')
    parser.add_argument('--path', help='테스트 이미지 경로')

    args = parser.parse_args()
    detection_image(args.path)
