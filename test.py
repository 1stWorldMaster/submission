import numpy as np
from PIL import Image

import cv2
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torchreid
import const

from torch.cuda.amp import autocast
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
from transformers import SwinModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detr_model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
detr_model.to(device)

if torch.cuda.is_available():
    detr_model = detr_model.half()
detr_model.eval()

detr_transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], 
                [0.229, 0.224, 0.225])
])

osnet_model = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
osnet_model.to(device)

if torch.cuda.is_available():
    osnet_model = osnet_model.half()
osnet_model.eval()

osnet_transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

def first_inference(frame):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = detr_transform(img_pil).unsqueeze(0).to(device)
    if torch.cuda.is_available():
        img_t = img_t.half()
    
    with autocast():
        with torch.no_grad():
            outputs = detr_model(img_t)
    
    probabilities = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probabilities[:, 1] > const.person_conf
    bboxes_scaled = outputs['pred_boxes'][0, keep]
    
    boxes = bboxes_scaled * torch.tensor([img_w, img_h, img_w, img_h], device=device)
    boxes_top_left = boxes[:, :2] - boxes[:, 2:] / 2
    boxes_bottom_right = boxes[:, :2] + boxes[:, 2:] / 2
    
    boxes_xyxy = torch.cat([boxes_top_left, boxes_bottom_right], dim=1)
    
    return boxes_xyxy.cpu().numpy().astype(int)

def detr_model_output(frame, bbox):
    """
    Args
    ----
    frame : np.ndarray (BGR)
    bbox  : iterable of length-4, (x1, y1, x2, y2) in the *same image space*
            All detections whose centroids lie outside this box are discarded.
    """
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t   = detr_transform(img_pil).unsqueeze(0).to(device)

    if torch.cuda.is_available():
        img_t = img_t.half()

    with autocast(), torch.no_grad():
        outputs = detr_model(img_t)

    # -- keep the person class only (unchanged) -------------------------------
    prob = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = prob[:, 1] > const.person_conf
    boxes = outputs['pred_boxes'][0, keep]

    # -- convert cxcywh → xyxy in absolute pixels ----------------------------
    img_h, img_w = frame.shape[:2]
    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], device=device)
    tl    = boxes[:, :2] - boxes[:, 2:] / 2          # top-left  (x1,y1)
    br    = boxes[:, :2] + boxes[:, 2:] / 2          # bottom-rt (x2,y2)
    boxes_xyxy = torch.cat([tl, br], dim=1)          # [N,4], xyxy

    # -- keep only those whose centroid is inside `bbox` ----------------------
    # convert bbox to tensor on same device
    bbox_t = torch.tensor(bbox, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)

    # centroids
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0

    inside = (
        (cx >= bbox_t[0]) & (cx <= bbox_t[2])
        & (cy >= bbox_t[1]) & (cy <= bbox_t[3])
    )

    boxes_xyxy = boxes_xyxy[inside]                  # filter in-place

    return boxes_xyxy.cpu().numpy().astype(int)

def osnet_x1_model_output(frame, boxes_xyxy):
    if len(boxes_xyxy) == 0:
        return []

    img_tensors = []
    for x_min, y_min, x_max, y_max in boxes_xyxy:
        crop = frame[y_min:y_max, x_min:x_max]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(crop)  
        img_tensors.append(osnet_transform(crop).to(device))

    img_tensors = torch.stack(img_tensors).to(device)
    if torch.cuda.is_available():
        img_tensors = img_tensors.half()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        features = osnet_model(img_tensors)

    return [[box, feat.cpu().numpy()] for box, feat in zip(boxes_xyxy, features)]

def select_person(frame, boxes):
    if frame is None:
        raise ValueError("Provided frame is None.")

    boxes = [tuple(map(int, b)) for b in boxes]
    selected_idx = [None]

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_idx[0] = i
                    break

    cv2.namedWindow('Select Box')
    cv2.setMouseCallback('Select Box', mouse_cb)

    while True:
        disp = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = (0, 255, 0) if i == selected_idx[0] else (255, 0, 0)
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        cv2.imshow('Select Box', disp)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()
    return selected_idx[0]

def mark_person(person_embeddings, person_feature, threshold=0.7):
    for box, emb in person_feature:
        sim = np.dot(person_embeddings, emb) / (
            np.linalg.norm(person_embeddings) * np.linalg.norm(emb)
        )
        if sim >= threshold:
            return [box, emb]
    return None

def draw_box(frame, box):
    xmin, ymin, xmax, ymax = [int(v.item()) if torch.is_tensor(v) else int(v) for v in box]
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)


cap = cv2.VideoCapture(const.video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_interval = max(1, int(21/ const.fps))

frame_count = 0
second = 0
bbox = [572, 552, 1696, 981]
total_time = 0
last_time = 0
calc_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        draw_box(frame, bbox)
    
        # print(person_feature)


        if frame_count == 0:
            boxes_xyxy = first_inference(frame)
            person_feature = osnet_x1_model_output(frame, boxes_xyxy)
            index = select_person(frame, boxes_xyxy)
            track_feature = person_feature[index]
    
        else:
            boxes_xyxy = detr_model_output(frame,bbox)
            person_feature = osnet_x1_model_output(frame, boxes_xyxy)
            if person_feature is not None:
                track_feature =  mark_person(updated_embedding, person_feature) 

        if track_feature is not None:
            box , updated_embedding = track_feature
            draw_box(frame, box)

            calc_time = second - last_time
            if calc_time < const.buffer_time:
                total_time += calc_time
            last_time = second
            



        # if track_feature == None:
        #     print(f"may be working{second}")


        cv2.imshow("frames", frame)

        if cv2.waitKey(const.cv_wait) & 0xFF == ord('q'):
            break

        second+=1
    frame_count+=1
cap.release()
cv2.destroyAllWindows()

