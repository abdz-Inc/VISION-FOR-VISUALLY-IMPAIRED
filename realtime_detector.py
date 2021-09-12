import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import colorsys
from tensorflow.keras.layers import Input
from yad2k.models.keras_yolo import yolo_body
import cv2
import pyttsx3
import threading as T

#speech alert
engine = pyttsx3.init()
engine.setProperty("rate", 130)
voices = engine.getProperty("voices")
voice_id = voices[1].id
engine.setProperty("voice", voice_id)
#90deg
#0.753173828125 80.5 308.5
'''
speech_queue = []
template = "Alert {vehicle} ahead"

area_dict = {
    'bicycle':0.75,
    'person':0.4,
    'chair' : 0.27
}
'''

model = tf.keras.models.load_model("model_data\\", compile = False)

def yolo_head(outputs, anchors, classes):

    num_anch = len(anchors)
    anch_tens = K.reshape(K.variable(anchors), [1,1,1,num_anch, 2])
    #[[[[[5],[5]],[[5],[5]],[[5],[5]],[[5],[5]],[[5],[5]]]]]
    num_classes = len(classes)
    
    conv_dims = K.shape(outputs)[1:3]

    h8_i = K.arange(0, stop = conv_dims[0])
    h8_i = K.tile(h8_i,[conv_dims[0]])

    wth_i = K.arange(0, stop = conv_dims[1])
    wth_i = K.tile(K.expand_dims(wth_i, 0), [conv_dims[1],1])
    wth_i = K.flatten(K.transpose(wth_i))

    conv_i = K.transpose(K.stack([h8_i, wth_i]))
    conv_i = K.cast(conv_i, dtype = K.dtype(K.constant(outputs)))

    conv_i = K.reshape(conv_i, [1, conv_dims[0], conv_dims[1], 1, 2])

    outputs = K.reshape(outputs, [-1, conv_dims[0], conv_dims[1], num_anch, 5 + num_classes])

    conv_dims = K.cast(K.reshape(conv_dims, [1,1,1,1,2]), K.dtype(K.constant(outputs)))

    #get boxes
    box_xy = K.sigmoid(outputs[...,:2])
    box_wh = K.exp(outputs[...,2:4])
    box_conf = K.sigmoid(outputs[...,4:5])
    box_classes = K.softmax(outputs[...,5:])
    #print('box_conf',box_conf.shape)
    #print('box_class',box_classes.shape)

    box_xy = (box_xy+conv_i)/conv_dims
    box_wh = (box_wh * anch_tens)/conv_dims

    return box_xy, box_wh, box_conf, box_classes


def box_to_corners(box_xy, box_wh):

    box_min = box_xy - (box_wh/2)
    box_max = box_xy + (box_wh/2)

    return K.concatenate([
        box_min[...,1:2],#ymin
        box_min[...,0:1],#xmin
        box_max[...,1:2],#ymax
        box_max[...,0:1]
    ])


def yolo_filter_boxes(boxes, box_conf, box_cls_prob, threshold = 0.6):

    box_scores = box_conf*box_cls_prob
    
    box_classes = tf.math.argmax(box_scores, axis = -1)
    box_cls_scores = tf.reduce_max(box_scores, axis = -1)

    filter = box_cls_scores >= threshold

    boxes = tf.boolean_mask(boxes, filter)
    scores = tf.boolean_mask(box_cls_scores, filter)
    classes = tf.boolean_mask(box_classes, filter)
    #print(boxes)
    return boxes, scores, classes


def scale_boxes(boxes, image_size):

    h = image_size[0]*1.0
    w = image_size[1]*1.0

    scale = K.stack([h, w, h, w])
    scale = K.reshape(scale, [1,4])

    boxes = boxes*scale

    return boxes


def nms(boxes, scores, classes, iou_threshold = 0.5, max_out = 10):

    max_out = tf.Variable(max_out)

    indices = tf.image.non_max_suppression(boxes, scores, max_out, iou_threshold)

    boxes = tf.gather(boxes, indices)
    scores = tf.gather(scores,indices)
    classes = tf.gather(classes,indices)
    #print(classes)
    return boxes, scores, classes


def get_colors(num_classes):

    if hasattr(get_colors, "colors") and len(get_colors.colors) == num_classes:
        return get_colors.colors

    
    hsv_val = [(x/num_classes, 1., 1.) for x in range(num_classes)]
    rgbs = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_val))
    rgb = list(map(lambda x: (int(x[0])*255,int(x[1])*255,int(x[2])*255), rgbs))

    get_colors.colors = rgb
    return get_colors.colors


def draw_boxes(image, boxes, box_classes, classes, scores = None):

    #font = ImageFont.truetype(font = "font\\FiraMono-Medium.ttf", size=np.floor(3e-2*image.size[1]+0.5).astype('int32'))
    #print(image.size)
    #1 width 0 h8
    thickness = (image.shape[1]+image.shape[0])//300
    colors = get_colors(len(classes))

    for i, c in enumerate(box_classes):

        box = boxes[i]
        box_cls = classes[c]

        if isinstance(scores.numpy(), np.ndarray):
            label = f"{box_cls} : {scores[i]}"

        else:
            label = f"{box_cls}"
        
        #draw = ImageDraw.Draw(image)
        label_size = len(label)*np.floor(0.5*image.shape[1]+0.5)

        top, left, bottom, right = box
        top = max(0, np.floor(top+0.5).astype('int32'))
        left = max(0, np.floor(left+0.5).astype('int32'))
        bottom = min(image.shape[0], np.floor(bottom+0.5).astype('int32'))
        right = min(image.shape[1], np.floor(right+0.5).astype('int32'))

        if top-label_size >=0:
            text_orig = np.array([left, top-label_size])

        else:
            text_orig = np.array([left, top+1])

        img_area = image.shape[0]*image.shape[1]
        area = ((bottom-top)*(right-left))/img_area
        
        centers = image.shape[1]/2 - (left + abs(right - left)/2)

        '''
        for obj in classes:
            
            try:
                ar = area_dict[obj]
            except:
                ar = 0
            if box_cls == obj and area >= ar and centers<= abs(right - left)/2:

                if template.format(vehicle = obj) not in speech_queue:
                    speech_queue.append(template.format(vehicle = obj))
                    break
        '''
        print(box_cls,area , centers , (right - left)/2)
        if box_cls == 'person':
            #print()
            if area >= area_dict['person'] and abs(centers) <= abs(right - left)/2:
                if centers >= 0:
                    dir = 'right'
                else:
                    dir = 'left'
                if template.format(vehicle = 'person', direction = dir) not in speech_queue:
                    speech_queue.append(template.format(vehicle = 'person', direction = dir))

        elif box_cls == 'bicycle':
            if area >= area_dict['bicycle'] and abs(centers)<= abs(right - left)/2:
                if centers >= 0:
                    dir = 'right'
                else:
                    dir = 'left'
                if template.format(vehicle = 'bicycle', direction = dir) not in speech_queue:

                    speech_queue.append(template.format(vehicle = 'bicycle', direction = dir))

        elif box_cls == 'chair':
            if area >= area_dict['chair'] and abs(centers)<= abs(right - left)/2:
                if centers >= 0:
                    dir = 'right'
                else:
                    dir = 'left'
                if template.format(vehicle = 'chair', direction = dir) not in speech_queue:
                    speech_queue.append(template.format(vehicle = 'chair', direction = dir))


        '''for j in range(thickness):
            draw.rectangle([left+i, top+i, right-i, bottom-i], outline = colors[c])'''
        
        image = cv2.rectangle(image,(left,top), (right, bottom), color = colors[c], thickness=thickness)
        #draw.rectangle([tuple(text_orig), tuple(text_orig+label_size)], fill = colors[c])
        image = cv2.putText(image, label, text_orig, cv2.FONT_HERSHEY_PLAIN, 1, colors[c])
        #draw.text(text_orig, label, font = font, color=(0,0,0))
        #image = cv2.flip(image, 1)

    return np.array(image)[::-1]

def speaker(speech_queue, speaking):
    speaking = 1
    for i in speech_queue:
        engine.say(i)
        speech_queue.remove(i)
    engine.runAndWait()
    speaking = 0

    speech_queue = []


def detect_image(image):
    """
    image : Imge object
    """
    t1 = None
    img1 = cv2.resize(image, (608,608))
    img= np.asarray(img1)
    img = img/255
    img = np.reshape(img,(1,608,608,3))

    pred = model(img)
    
    with open('model_data\\yolo_anchors.txt', 'r') as f:

        anchors = f.readlines()
        anchors = [float(i) for i in anchors[0].split(',')]
        anchors = np.array(anchors).reshape(-1,2)
        #print(anchors)

    with open('model_data\\coco_classes.txt', 'r') as f:

        classes = f.readlines()
        classes = [i.strip() for i in classes]
        #print(classes)


    box_xy, box_wh, box_conf, box_classes = yolo_head(pred, anchors, classes)
    
    boxes =box_to_corners(box_xy, box_wh)

    boxes, scores, b_classes = yolo_filter_boxes(boxes, box_conf, box_classes, threshold = 0.6)
    
    boxes = scale_boxes(boxes, (image.shape[0], image.shape[1]))
    
    boxes, scores, b_classes = nms(boxes, scores, b_classes, iou_threshold = 0.5, max_out = 10)

    draw_boxes(image, boxes, b_classes, classes, scores)

    print(speech_queue)
    
    if speaking == 0 and t1 == None and speech_queue != []:
        t1 = T.Thread(target=speaker, args=(speech_queue, speaking))
        t1.start()

    if t1 != None and t1.is_alive():
        del t1
    
    return image


def detect_video(video_path = 0):
    import cv2
    print("about to open....")
    vid = cv2.VideoCapture(video_path)
    print("opened...")
    vid.set(1920,1080)
    print("set...")
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    '''video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()'''
    #print("enter loop")
    with tf.device('/GPU:0'):
        
        while True:
            return_value, frame = vid.read()
            #print("got frame")
            #image = Image.fromarray(frame)
            image = detect_image(frame)
            #print("detected")
            result = np.asarray(image)
            '''curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0'''
            #cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            #print("showing...")
            cv2.imshow("result", result)
            '''if isOutput:
                out.write(result)'''
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    engine = pyttsx3.init()
    speech_queue = []
    template = "Alert {vehicle} ahead. Move {direction}"
    speaking = 0
    area_dict = {
        'bicycle':0.73,
        'person':0.5,
        'chair' : 0.27
    }
    detect_video()