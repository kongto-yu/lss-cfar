from ultralytics import YOLO
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

def yolov8_detection(model, image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image, stream=True)  # generator of Results objects

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs

    bbox = boxes.xyxy.tolist()
    bbox = [[int(i) for i in box] for box in bbox]
    return bbox, image


model = YOLO("/home/lucayu/lss-cfar/yolov8_sam/pretrained_model/yolov8s.pt")
yolov8_boxex, image = yolov8_detection(model, "/home/lucayu/lss-cfar/yolov8_sam/test.png")
input_boxes = torch.tensor(yolov8_boxex, device=model.device)

sam_checkpoint = "/home/lucayu/lss-cfar/yolov8_sam/pretrained_model/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

for i, mask in enumerate(masks):

    binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)

    # Find the contours of the mask
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the new bounding box
    bbox = [int(x) for x in cv2.boundingRect(largest_contour)]

    # Get the segmentation mask for object
    segmentation = largest_contour.flatten().tolist()

    # Write bounding boxes to file in YOLO format
    with open("BBOX_yolo.txt", "w") as f:
        for contour in contours:
            # Get the bounding box coordinates of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Convert the coordinates to YOLO format and write to file
            f.write(
                "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    (x + w / 2) / image.shape[1],
                    (y + h / 2) / image.shape[0],
                    w / image.shape[1],
                    h / image.shape[0],
                )
            )
            f.write("\n")
    mask = segmentation

    # load the image
    # width, height = image_path.size
    height, width = image.shape[:2]

    # convert mask to numpy array of shape (N,2)
    mask = np.array(mask).reshape(-1, 2)

    # normalize the pixel coordinates
    mask_norm = mask / np.array([width, height])

    # compute the bounding box
    xmin, ymin = mask_norm.min(axis=0)
    xmax, ymax = mask_norm.max(axis=0)
    bbox_norm = np.array([xmin, ymin, xmax, ymax])

    # concatenate bbox and mask to obtain YOLO format
    # yolo = np.concatenate([bbox_norm, mask_norm.reshape(-1)])
    yolo = mask_norm.reshape(-1)

    # compute the bounding box
    # write the yolo values to a text file
    with open("yolomask_format.txt", "a") as f:
        for val in yolo:
            f.write("{:.6f} ".format(val))
        f.write("\n")

    print("Bounding box:", bbox)
    print("yolo", yolo)