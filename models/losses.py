import torch
from torch import nn
import torchvision
import numpy as np
from scipy.ndimage import label

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(torch.nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()#: changed to be able to run on gpu
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class ConnectivityLoss(nn.Module):
    """
    Penalizes topological errors (False Merges & Splits) by weighting the BCE loss.
    Based on: 'Few-Shot Connectivity-Aware Text Line Segmentation in Historical Documents'
    """
    def __init__(self, alpha=1.0, beta=0.0, penalty_weight=10.0):
        super(ConnectivityLoss, self).__init__()
        self.alpha = alpha   # 0.0 = Off, 1.0 = Max Topology Focus
        self.beta = beta     # 0.0 = Penalize Merges, 1.0 = Penalize Splits
        self.penalty = penalty_weight # How much to boost loss on error pixels
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def get_component_masks(self, binary_img):
        """Returns labeled components from a binary mask (CPU/Numpy)."""
        labeled_array, num_features = label(binary_img)
        return labeled_array, num_features

    def forward(self, pred, target):
        """
        pred: (B, C, H, W) Output from model (Logits or [-1, 1])
        target: (B, C, H, W) Ground Truth [-1, 1]
        """
        # Normalize Inputs
        # Target: [-1, 1] -> [0, 1]
        target_01 = (target + 1) / 2.0
        
        # Pred: Handle Logits vs Tanh output
        if pred.min() < 0:
            pred_prob = (pred + 1) / 2.0 # Assume Tanh -> [0, 1]
        else:
            pred_prob = pred

        # Base Pixel Loss (L0)
        # We calculate the standard loss for every pixel first
        # We use L1 here to match your model's style, or BCE
        pixel_loss = torch.abs(pred_prob - target_01) 

        # If alpha is 0 (or during early training), skip expensive calculation
        if self.alpha == 0:
            return pixel_loss.mean()

        # Topology Weight Map Calculation
        # We create a map of "1.0"s, and add "10.0" (penalty) to bad pixels
        weight_map = torch.ones_like(pred)
        
        # Move to CPU for Connectivity Analysis
        pred_mask_np = (pred_prob.detach().cpu().numpy() > 0.5).astype(np.uint8)
        target_mask_np = (target_01.detach().cpu().numpy() > 0.5).astype(np.uint8)
        
        batch_size = pred.shape[0]
        
        for b in range(batch_size):
            p_mask = pred_mask_np[b, 0]
            t_mask = target_mask_np[b, 0]
            
            # --- DETECT FALSE MERGES (Beta < 0.5) ---
            # Paper says Beta=0 (Merges) worked best for historical docs
            if self.beta < 0.5:
                p_labels, p_num = self.get_component_masks(p_mask)
                t_labels, t_num = self.get_component_masks(t_mask)
                
                if p_num > 0 and t_num > 1:
                    for i in range(1, p_num + 1):
                        # Get mask of one predicted blob
                        comp_mask = (p_labels == i)
                        # Check how many GROUND TRUTH lines it touches
                        touched = np.unique(t_labels[comp_mask])
                        touched = touched[touched != 0] # Ignore background
                        
                        # If it touches > 1 target line, it is a MERGE ERROR
                        if len(touched) > 1:
                            # Penalize the 'Bridge' pixels (Pred=1, Target=0)
                            error_pixels = comp_mask & (t_mask == 0)
                            weight_map[b, 0][error_pixels] += self.penalty

            # --- DETECT FALSE SPLITS (Beta > 0.5) ---
            if self.beta > 0.5:
                t_labels, t_num = self.get_component_masks(t_mask)
                p_labels, p_num = self.get_component_masks(p_mask)
                
                if t_num > 0:
                    for i in range(1, t_num + 1):
                        comp_mask = (t_labels == i)
                        # Check if this single GT line is broken in Pred
                        pred_in_target = p_mask * comp_mask
                        sub_labels, sub_num = self.get_component_masks(pred_in_target)
                        
                        if sub_num > 1:
                            # Penalize the Gaps (Target=1, Pred=0)
                            error_pixels = comp_mask & (p_mask == 0)
                            weight_map[b, 0][error_pixels] += self.penalty

        # Apply Weighted Loss
        # Formula: (1 - alpha) * L0 + alpha * Weighted_L0
        weighted_loss = (pixel_loss * weight_map).mean()
        
        return (1 - self.alpha) * pixel_loss.mean() + self.alpha * weighted_loss
