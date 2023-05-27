import cv2
import torch
import numpy as np
import gradio as gr
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from segment_anything import sam_model_registry, SamPredictor

models = {
	'vit_h': './pretrained/sam_vit_h_4b8939.pth'
}

vitmatte_models = {
	'vit_b': './pretrained/ViTMatte_B_DIS.pth',
}

vitmatte_config = {
	'vit_b': './configs/matte_anything.py',
}

def init_segment_anything(model_type):
    """
    Initialize the segmenting anything with model_type in ['vit_b', 'vit_l', 'vit_h']
    """
    
    sam = sam_model_registry[model_type](checkpoint=models[model_type]).to(device)
    predictor = SamPredictor(sam)

    return predictor

def init_vitmatte(model_type):
    """
    Initialize the vitmatte with model_type in ['vit_s', 'vit_b']
    """
    cfg = LazyConfig.load(vitmatte_config[model_type])
    vitmatte = instantiate(cfg.model)
    vitmatte.to(device)
    vitmatte.eval()
    DetectionCheckpointer(vitmatte).load(vitmatte_models[model_type])

    return vitmatte

def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    eroded = cv2.erode(mask, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
    trimap = np.zeros_like(mask)
    trimap[dilated==255] = 128
    trimap[eroded==255] = 255
    return trimap

# user click the image to get points, and show the points on the image
def get_point(img, sel_pix, point_type, evt: gr.SelectData):
    if point_type == 'foreground_point':
        sel_pix.append((evt.index, 1))   # append the foreground_point
    elif point_type == 'background_point':
        sel_pix.append((evt.index, 0))    # append the background_point
    else:
        sel_pix.append((evt.index, 1))    # default foreground_point
    # draw points
    for point, label in sel_pix:
        cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img if isinstance(img, np.ndarray) else np.array(img)

# undo the selected point
def undo_points(orig_img, sel_pix):
    temp = orig_img.copy()
    # draw points
    if len(sel_pix) != 0:
        sel_pix.pop()
        for point, label in sel_pix:
            cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
    if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    return temp if isinstance(temp, np.ndarray) else np.array(temp)

# once user upload an image, the original image is stored in `original_image`
def store_img(img):
    return img, []  # when new image is uploaded, `selected_points` should be empty

if __name__ == "__main__":
    device = 'cuda'
    sam_model = 'vit_h'
    vitmatte_model = 'vit_b'
    
    colors = [(255, 0, 0), (0, 255, 0)]
    markers = [1, 5]

    print('Initializing models... Please wait...')

    predictor = init_segment_anything(sam_model)
    vitmatte = init_vitmatte(vitmatte_model)

    def run_inference(input_x, selected_points):
        predictor.set_image(input_x)
        if len(selected_points) != 0:
            points = torch.Tensor([p for p, _ in selected_points]).to(device).unsqueeze(1)
            labels = torch.Tensor([int(l) for _, l in selected_points]).to(device).unsqueeze(1)
            transformed_points = predictor.transform.apply_coords_torch(points, input_x.shape[:2])
            print(points.size(), transformed_points.size(), labels.size(), input_x.shape, points)
        else:
            transformed_points, labels = None, None
                    
        # predict segmentation according to the boxes
        masks, scores, logits = predictor.predict_torch(
            point_coords=transformed_points.permute(1, 0, 2),
            point_labels=labels.permute(1, 0),
            boxes=None,
            multimask_output=False,
        )
        masks = masks.cpu().detach().numpy()
        mask_all = np.ones((input_x.shape[0], input_x.shape[1], 3))
        for ann in masks:
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                mask_all[ann[0] == True, i] = color_mask[i]
        img = input_x / 255 * 0.3 + mask_all * 0.7
        
        # generate alpha matte
        torch.cuda.empty_cache()
        mask = masks[0][0].astype(np.uint8)*255
        trimap = generate_trimap(mask).astype(np.float32)
        trimap[trimap==128] = 0.5
        trimap[trimap==255] = 1

        input = {
            "image": torch.from_numpy(input_x).permute(2, 0, 1).unsqueeze(0)/255,
            "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
        }

        alpha = vitmatte(input)['phas'].flatten(0,2)
        alpha = alpha.detach().cpu().numpy()
        
        # get a green background
        background = np.ones_like(input_x) * np.array([0, 254, 0])

        # calculate foreground with alpha blending
        foreground_alpha = input_x * np.expand_dims(alpha, axis=2).repeat(3,2)/255

        # calculate foreground with mask
        foreground_mask = input_x * np.expand_dims(mask/255, axis=2).repeat(3,2)/255

        # return img, mask_all
        trimap[trimap==1] == 0.999
        return mask, foreground_mask, foreground_alpha

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # <center>Matte Anything🐒 !
            """
        )
        with gr.Row().style(equal_height=True):
            with gr.Column():
                # input image
                original_image = gr.State(value=None)   # store original image without points, default None
                input_image = gr.Image(type="numpy")
                # point prompt
                with gr.Column():
                    selected_points = gr.State([])      # store points
                    with gr.Row():
                        undo_button = gr.Button('Remove Points')
                    radio = gr.Radio(['foreground_point', 'background_point'], label='point labels')
                # run button
                button = gr.Button("Start!")
            # show the image with mask
            with gr.Tab(label='SAM Mask'):
                mask = gr.Image(type='numpy')
            # with gr.Tab(label='Trimap'):
            #     trimap = gr.Image(type='numpy')
            # show only mask
            with gr.Tab(label='Foreground by SAM Mask'):
                foreground_by_sam_mask = gr.Image(type='numpy')
            with gr.Tab(label='Refined by ViTMatte'):
                refined_by_vitmatte = gr.Image(type='numpy')
        input_image.upload(
            store_img,
            [input_image],
            [original_image, selected_points]
        )
        input_image.select(
            get_point,
            [input_image, selected_points, radio],
            [input_image],
        )
        undo_button.click(
            undo_points,
            [original_image, selected_points],
            [input_image]
        )
        button.click(run_inference, inputs=[original_image, selected_points], outputs=[mask, foreground_by_sam_mask, refined_by_vitmatte])

    demo.launch()