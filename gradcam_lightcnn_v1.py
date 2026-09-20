# -*- coding: utf-8 -*-
"""Grad-CAM utilities for SinusitisLightCNN.
Target layer: the last Conv2d layer, discovered automatically.
No training, random augmentation, or image flipping is performed.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model_definition_lightcnn_v2 import SinusitisLightCNN, CLASS_ORDER


def load_checkpoint_model(checkpoint_path, device='cpu'):
    device = torch.device(device)
    try:
        obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch 1.12.x
        obj = torch.load(checkpoint_path, map_location=device)

    state = obj
    if isinstance(obj, dict):
        for key in ('model', 'model_state_dict', 'state_dict'):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    if not isinstance(state, dict):
        raise TypeError('No state_dict was found in the checkpoint.')
    if state and all(str(k).startswith('module.') for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}

    model = SinusitisLightCNN().to(device)
    result = model.load_state_dict(state, strict=True)
    model.eval()
    return model, result


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((96, 96), Image.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return rgb, tensor


def last_conv_layer(model):
    layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not layers:
        raise RuntimeError('No Conv2d layer was found.')
    return layers[-1]


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or last_conv_layer(model)
        self.activations = None
        self.gradients = None
        self.handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __call__(self, image_tensor, class_index=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted = int(probabilities.argmax(dim=1).item())
        target = predicted if class_index is None else int(class_index)
        if target < 0 or target >= len(CLASS_ORDER):
            raise ValueError('class_index must be 0, 1, 2, or 3.')
        logits[0, target].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError('Grad-CAM hooks did not capture activations/gradients.')
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(96, 96), mode='bilinear', align_corners=False)
        cam = cam[0, 0]
        cmin, cmax = cam.min(), cam.max()
        if float(cmax - cmin) > 1e-12:
            cam = (cam - cmin) / (cmax - cmin)
        else:
            cam = torch.zeros_like(cam)
        return cam.cpu().numpy(), probabilities[0].detach().cpu().numpy(), predicted, target


def save_gradcam_figure(rgb, heatmap, probabilities, predicted, target, output_path,
                        true_class=None, alpha=0.40):
    cmap = plt.get_cmap('jet')
    colored = cmap(heatmap)[..., :3]
    overlay = np.clip((1 - alpha) * rgb + alpha * colored, 0, 1)
    true_text = 'NA' if true_class is None else CLASS_ORDER[int(true_class)]
    title = f'True: {true_text} | Pred: {CLASS_ORDER[predicted]} | CAM target: {CLASS_ORDER[target]}'

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb); axes[0].set_title('Input')
    axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1); axes[1].set_title('Grad-CAM')
    axes[2].imshow(overlay); axes[2].set_title(title, fontsize=9)
    for ax in axes: ax.axis('off')
    prob_text = ' | '.join(f'{CLASS_ORDER[i]} {probabilities[i]*100:.1f}%' for i in range(4))
    fig.suptitle(prob_text, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return overlay
