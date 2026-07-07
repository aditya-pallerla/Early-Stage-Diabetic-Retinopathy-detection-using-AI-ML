# edi-backend/utils/gradcam_service.py (FOR IMAGE-ONLY MODEL)

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import base64

# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================
class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()
    
    def hook_layers(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # Find last Conv2d in the target layer
        if isinstance(self.target_layer, torch.nn.Sequential):
            last_conv = None
            for module in reversed(list(self.target_layer.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
                    break
            
            if last_conv is None:
                raise ValueError("No Conv2d found in target layer")
            
            last_conv.register_forward_hook(forward_hook)
            last_conv.register_full_backward_hook(backward_hook)
        else:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_image, class_idx=None):
        """
        Generate GradCAM heatmap
        
        Args:
            input_image: Preprocessed image tensor
            class_idx: Target class (default: predicted class)
        
        Returns:
            cam: 2D numpy array with heatmap
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        # Backward pass
        output[:, class_idx].backward()
        
        gradients = self.gradients  # [batch, channels, h, w]
        activations = self.activations  # [batch, channels, h, w]
        
        # Compute weights via global average pooling
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1).squeeze().cpu().numpy()
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        return cam

# ============================================================================
# GRAD-CAM GENERATION FOR IMAGE-ONLY MODEL
# ============================================================================
def get_gradcam_base64(image_model, image_bytes, device, img_size=(512, 512)):
    """
    Generate GradCAM from image-only model and return as base64
    
    Args:
        image_model: ImageOnlyNet (EfficientNet-B4)
        image_bytes: Raw image bytes
        device: torch device
        img_size: Image size
    
    Returns:
        Base64-encoded GradCAM overlay image
    """
    # ========== PREPROCESS IMAGE ==========
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # ========== SELECT TARGET LAYER ==========
    # For EfficientNet-B4, use last block (block 6)
    target_layer = image_model.backbone.blocks[6]
    
    # ========== GENERATE CAM ==========
    gradcam = GradCAM(image_model, target_layer)
    cam_raw = gradcam.generate(input_tensor, class_idx=1)  # Focus on Stage 1
    
    # ========== POST-PROCESS CAM ==========
    # Resize to match input image
    cam = cv2.resize(cam_raw, (img_size[1], img_size[0]))
    
    # Normalize to [0, 1]
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    
    # Apply JET colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # ========== OVERLAY ON ORIGINAL IMAGE ==========
    img_np = np.array(image.resize((img_size[1], img_size[0])))
    
    # Blend: 40% heatmap + 60% original
    overlay = np.float32(heatmap) * 0.4 + np.float32(img_np) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # ========== CONVERT TO BASE64 ==========
    final_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    final_img.save(buffer, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_base64}"


# ============================================================================
# OPTIONAL: SIDE-BY-SIDE COMPARISON
# ============================================================================
def get_gradcam_comparison(image_model, image_bytes, device, img_size=(512, 512)):
    """
    Generate side-by-side: Original | Heatmap | Overlay
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    img_np = np.array(image.resize((img_size[1], img_size[0])))
    
    # Generate CAM
    target_layer = image_model.backbone.blocks[6]
    gradcam = GradCAM(image_model, target_layer)
    cam_raw = gradcam.generate(input_tensor, class_idx=1)
    cam = cv2.resize(cam_raw, (img_size[1], img_size[0]))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.float32(heatmap) * 0.4 + np.float32(img_np) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Create comparison
    comparison = np.hstack([img_np, heatmap, overlay])
    
    # Convert to base64
    comparison_img = Image.fromarray(comparison)
    buffer = io.BytesIO()
    comparison_img.save(buffer, format="JPEG", quality=95)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_base64}"