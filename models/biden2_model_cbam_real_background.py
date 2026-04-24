import random
import torch.nn.functional as F
import torch
from .base_model import BaseModel
from . import networks
#from .losses import VGGLoss
from .losses import VGGLoss, ConnectivityLoss
from numpy import *
import itertools
from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
import skimage as ski
from .utils_blindsr import add_blur, add_resize, add_Gaussian_noise, add_speckle_noise, add_Poisson_noise
from .architecture_utils import ContextAwareDecoder 


""" 
BIDeN model for Task I, Mixed image decomposition across multiple domains. Max number of domain = 2.

Sample usage:
Optional visualization:
python -m visdom.server

Train:
python train.py --dataroot ./datasets/image_decom --name biden2 --model biden2 --dataset_mode unaligned2

For test:
Test a single case:
python test.py --dataroot ./datasets/image_decom --name biden2 --model biden2 --dataset_mode unaligned2 --test_input A
python test.py --dataroot ./datasets/image_decom --name biden2 --model biden2 --dataset_mode unaligned2 --test_input AB
... ane other cases.
change test_input to the case you want.

Test all cases:
python test2.py --dataroot ./datasets/image_decom --name biden2 --model biden2 --dataset_mode unaligned2
"""

class BIDEN2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for BIDeN model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        parser.add_argument('--lambda_Ln', type=float, default=30.0, help='weight for L1/L2 loss') 
        parser.add_argument('--lambda_VGG', type=float, default=10.0, help='weight for VGG loss')
        parser.add_argument('--lambda_BCE', type=float, default=1.0, help='weight for BCE loss')
        parser.add_argument('--test_input', type=str, default='A', help='test mixed images.')
        parser.add_argument('--max_domain', type=int, default=2, help='max number of source components.')
        parser.add_argument('--prob', type=float, default=0.9, help='probability of adding a component')
        parser.add_argument('--test_choice', type=int, default=1, help='choice for test mode, 1 for one case,'
                                                                       ' 0 for all cases. Will be set automatically.')
        parser.add_argument('--pre_mixed', action='store_true', help='Use this if input A is already a mixed image. Skips synthetic mixing.')
        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'Ln', 'VGG', 'BCE']
        self.visual_names = ['fake_A', 'fake_B', 'real_A', 'real_B', 'real_input']
        self.model_names = ['D', 'E', 'H1', 'H2']

        # Define networks (both generator and discriminator)
        # Define Encoder E.
        self.netE = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'encoder', opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                      opt.no_antialias_up, self.gpu_ids, opt)
        # Define Heads H.
        self.netH1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'head', opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       opt.no_antialias_up, self.gpu_ids, opt)
        
        # Original definition of H2 (similar to H1
        #self.netH2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'head', opt.normG,
                                       #not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                       #opt.no_antialias_up, self.gpu_ids, opt)

        # Define netH2 (Undertext - UPGRADED)
        # We replace the generic define_G with our specialized class
        encoder_out_dim = opt.ngf * 8
            
        self.netH2 = ContextAwareDecoder(encoder_out_dim, opt.output_nc, opt.ngf).to(self.device)
        networks.init_net(self.netH2, opt.init_type, opt.init_gain, self.gpu_ids)
        
        self.label = torch.zeros(self.opt.max_domain).to(self.device)
        self.netD = networks.define_D(opt.output_nc, self.opt.max_domain, 'BIDeN_D', opt.n_layers_D, opt.normD,
                                      opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.correct = 0
        self.all_count = 0
        self.psnr_count = [0] * self.opt.max_domain
        self.acc_all = 0
        self.test_time = 0
        self.criterionL2 = torch.nn.MSELoss().to(self.device)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionVGG = VGGLoss(opt).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss().to(self.device)
            self.criterionConn = ConnectivityLoss(alpha=1.0, beta=0.0, penalty_weight=1.0).to(self.device) # connectivity preserving loss
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netE.parameters(), self.netH1.parameters(),
                                self.netH2.parameters()),
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def data_dependent_initialize(self, data):
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()


    def get_real_background(self, batch_size, height, width):
        """
        Loads real parchment patches.
       
        """
        import glob
        bg_paths = glob.glob('./datasets/background/*.jpg') 
        
        # Fallback to synthetic if no real images found
        if len(bg_paths) == 0:
            print("bg not found")
            return self.create_stained_background_torch(batch_size, height, width)
            
        real_bgs = []
        for _ in range(batch_size):
            # 1. Load Random Image
            path = random.choice(bg_paths)
            bg_pil = Image.open(path).convert("RGB")
            W_orig, H_orig = bg_pil.size
            
            # 2. Aggressive Geometric Augmentations
            
            # A. Random Flip (Horizontal & Vertical)
            if random.random() < 0.5:
                bg_pil = bg_pil.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                bg_pil = bg_pil.transpose(Image.FLIP_TOP_BOTTOM)
            
            # B. Random 90-degree Rotations (0, 90, 180, 270)
            
            rotations = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
            rot_choice = random.choice(rotations)
            if rot_choice:
                bg_pil = bg_pil.transpose(rot_choice)
                
            # C. Random Crop / Resize
            # If the image is large, pick a random patch
            if W_orig > width and H_orig > height:
                left = random.randint(0, W_orig - width)
                top = random.randint(0, H_orig - height)
                bg_pil = bg_pil.crop((left, top, left + width, top + height))
            else:
                # If image is small, Resize with random scale (Zoom in/out)
                # Scale between 1.0x and 1.5x (Zoom in slightly)
                scale = random.uniform(1.0, 1.5)
                new_w = int(width * scale)
                new_h = int(height * scale)
                bg_pil = bg_pil.resize((new_w, new_h), Image.BICUBIC)
                # Center crop after zoom
                left = (new_w - width) // 2
                top = (new_h - height) // 2
                bg_pil = bg_pil.crop((left, top, left + width, top + height))

            # 3. Convert to Tensor [0, 1]
            t = transforms.ToTensor()(bg_pil) 
            real_bgs.append(t)
            
        return torch.stack(real_bgs).to(self.device)

    
    def create_stained_background_torch(self, batch_size, height, width):
        """
        Generates parchment with realistic STAINS and CLOUDS (Low-frequency noise).
        """
    
        # We vary it slightly per batch
        base_r = (torch.rand(batch_size, 1, 1, 1, device=self.device) * 20.0 + 170.0) / 255.0
        base_g = (torch.rand(batch_size, 1, 1, 1, device=self.device) * 20.0 + 130.0) / 255.0
        base_b = (torch.rand(batch_size, 1, 1, 1, device=self.device) * 20.0 + 90.0) / 255.0
        
        bg = torch.cat([base_r, base_g, base_b], dim=1).expand(batch_size, 3, height, width)

        # Add "Stains" (Low Frequency Noise)
        stain_resolution = 16 
        small_noise = torch.rand(batch_size, 1, stain_resolution, stain_resolution, device=self.device)
        # Upsample to full size (creates cloudy look)
        large_stains = F.interpolate(small_noise, size=(height, width), mode='bilinear', align_corners=False)
        
        # Darken areas where stains are strong
        stain_intensity = 0.3 # How dark the stains are
        bg = bg - (large_stains * stain_intensity)
        
        # Add "Grain" (High Frequency Noise)
        grain = torch.randn(batch_size, 3, height, width, device=self.device) * 0.03
        bg = bg + grain
            
        return torch.clamp(bg, 0.0, 1.0)
    
    def create_parchment_background_torch(self, batch_size, height, width):
        """
        Generates parchment background on GPU.
        Randomly varies between Light Beige (Clean) and Dark Brown (Aged).
        """
        # Base Tone Variation (Subtle shifts in the beige/yellow hue)
        # Generate shapes (B, 1, 1, 1)
        base_r = (torch.rand(batch_size, 1, 1, 1, device=self.device) * 15.0 + 240.0) / 255.0
        base_g = (torch.rand(batch_size, 1, 1, 1, device=self.device) * 15.0 + 230.0) / 255.0
        base_b = (torch.rand(batch_size, 1, 1, 1, device=self.device) * 20.0 + 210.0) / 255.0
        
        # Combine into (B, 3, 1, 1)
        base_color = torch.cat([base_r, base_g, base_b], dim=1)

        # Random Aging (Darkness)
        # Random intensity (B, 1, 1, 1)
        intensity = torch.rand(batch_size, 1, 1, 1, device=self.device)
        max_age_drop = 90.0 / 255.0
        age_drop = intensity * max_age_drop
        
        # Apply Color + Texture
        # Texture (B, 3, H, W)
        texture = torch.randn(batch_size, 3, height, width, device=self.device) * 0.02
        
        # Broadcasting Logic: 
        # (B, 3, 1, 1) - (B, 1, 1, 1) + (B, 3, H, W) -> Result (B, 3, H, W)
        bg = base_color - age_drop + texture
            
        return torch.clamp(bg, 0.0, 1.0)

    def degrade_tensor(self, img_tensor):
        """
        Applies random noise and blur to a tensor (B, 3, H, W).
        """
        B, C, H, W = img_tensor.shape
        out = img_tensor.clone()
        
        # Gaussian Noise
        if random.random() < 0.8:
            # Random Noise Level: Sigma between 5/255 and 30/255
            # We use a random sigma per batch item (approximated by multiplying noise)
            noise_level = torch.rand(B, 1, 1, 1, device=self.device) * (25.0/255.0) + (5.0/255.0)
            gaussian_noise = torch.randn_like(out) * noise_level
            out = out + gaussian_noise

        # Speckle Noise
        if random.random() < 0.6:
            noise_level = torch.rand(B, 1, 1, 1, device=self.device) * (25.0/255.0) + (5.0/255.0)
            speckle = torch.randn_like(out) * noise_level
            out = out + (out * speckle)

        # Blur
        if random.random() < 0.7:
            # Random sigma between 0.5 and 2.5
            sigma = random.uniform(0.5, 2.5)
            # Use fixed kernel size 5 for GPU efficiency
            blurrer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=sigma)
            out = blurrer(out)

        return torch.clamp(out, 0.0, 1.0)
        
    def generate_liquid_ink(self, batch_size, H, W, profile="iron_gall"):
        """
        Generates a base ink color WITHOUT high-frequency RGB noise.
        Texture should come from the paper interaction, not the ink itself.
        """
        if profile == "iron_gall":
            # Iron Gall: Dark Brownish-Grey
            base_r = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.05 + 0.20
            base_g = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.05 + 0.15
            base_b = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.05 + 0.10
        else: 
            # Carbon/Dark: Near Black
            base_r = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.05
            base_g = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.05
            base_b = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.05

        base_color = torch.cat([base_r, base_g, base_b], dim=1)
        
        # Expand to full size
        ink_layer = base_color.expand(batch_size, 3, H, W)
        
        # Add very subtle low-frequency variation (pooling), NOT grain
        pooling = torch.randn(batch_size, 1, H // 4, W // 4, device=self.device)
        pooling = F.interpolate(pooling, size=(H, W), mode='bilinear') * 0.1
        
        return torch.clamp(ink_layer + pooling, 0.0, 1.0)
        
    
    def generate_granular_ink(self, batch_size, H, W, profile="iron_gall"):
        """
        Generates granular ink texture based on specific chemical profiles.
        Options: 'iron_gall', 'carbon', 'sepia', 'rust'
        """
        # Define Color Profiles (R, G, B ranges)
        if profile == "iron_gall":
            # The most common palimpsest ink. Fades to Grey-Brown.
            # R: 0.25-0.35, G: 0.20-0.30, B: 0.15-0.25 (Desaturated)
            base_r = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.25
            base_g = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.20
            base_b = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.15
            
        elif profile == "carbon":
            # Carbon black doesn't fade to brown; it stays Neutral Grey.
            val = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.15 + 0.25
            base_r = val
            base_g = val
            base_b = val + 0.02 # Slight blue tint sometimes

        elif profile == "rust":
            # Highly oxidized (very orange/red)
            base_r = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.40
            base_g = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.20
            base_b = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.05 + 0.05
            
        else: # "dark" (For Overtext)
            # Nearly Black / Dark Brown
            base_r = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.10
            base_g = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.10
            base_b = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.10

        base_color = torch.cat([base_r, base_g, base_b], dim=1)

        # Add Granularity (Pigment Noise)
        # Iron Gall ink is grainier than Carbon ink
        noise_scale = 0.2 if profile == "iron_gall" else 0.15
        
        # High-frequency grain
        grain = torch.randn(batch_size, 3, H, W, device=self.device) * noise_scale
        
        # Low-frequency pooling (Ink density variation)
        clumps = torch.randn(batch_size, 3, H // 8, W // 8, device=self.device)
        clumps = F.interpolate(clumps, size=(H, W), mode='bilinear') * 0.15

        # Combine
        final_ink = base_color + grain + clumps
        return torch.clamp(final_ink, 0.0, 1.0)
        
    def palimpsest_mix(self, A, B):
        """
        Mixes Uppertext (A) and Undertext (B) with realistic ink texture for BOTH layers.
        """
        # Map inputs from [-1, 1] to [0, 1]
        real_A = (A + 1) / 2.0
        real_B = (B + 1) / 2.0
        
        batch_size, _, H, W = real_A.shape

        # --- PREPARE UNDERTEXT (Faint, Sepia, Eroded) ---
        # Degrade B (Random noise/blur)
        B_degraded = self.degrade_tensor(real_B)
        
        # Soft Alpha Mask for B
        text_mask_B = 1.0 - B_degraded.mean(dim=1, keepdim=True)
        
        opacity_B = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.3 + 0.3
        alpha_B = torch.pow(text_mask_B.clamp(0, 1), 3.0) * opacity_B
        
        # --- HARD MODE: TEXT FRAGMENTATION 
        # Create a random noise mask (holes in the ink)
        # 40% of the ink pixels are randomly removed
        flaking_percentage=0.3 + 0.4 * torch.rand(1, device=text_mask_B.device)
        flaking_mask = (torch.rand_like(text_mask_B) > flaking_percentage).float()
        alpha_B = alpha_B * flaking_mask

        #ink_r = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.1 + 0.35 
        #ink_g = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.1 + 0.15 
        #ink_b = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.1 + 0.05
        #ink_color_B = torch.cat([ink_r, ink_g, ink_b], dim=1)
        
        ink_r = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.1 + 0.1 
        ink_g = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.1 + 0.1 
        ink_b = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.1 + 0.1
        ink_color_B = torch.cat([ink_r, ink_g, ink_b], dim=1)
        #ink_color_B = torch.zeros(batch_size, 3, 1, 1, device=self.device) # absolute black for binarized inputs

        # --- PREPARE UPPERTEXT ---
        # Add Texture to the Shape (Crucial for Realism)
        # Real ink isn't a flat shape. It has internal noise.
        # We add noise to the source image before calculating alpha.
        A_noisy = real_A + (torch.randn_like(real_A) * 0.1) 
        
        # Calculate Alpha with Variable Sharpness
        # Instead of fixed sharp edges, we vary the bleeding (sharpness)
        text_mask_A = 1.0 - A_noisy.mean(dim=1, keepdim=True)
        
        #sharpness = torch.rand(batch_size, 1, 1, 1, device=self.device) * 3.0 + 2.0
        #alpha_A = torch.pow(text_mask_A.clamp(0, 1), sharpness)
       
        sharpness = torch.rand(batch_size, 1, 1, 1, device=self.device) * 1.5 + 1.5
        opacity_A = 1.0 #torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.25 + 0.80 # 1.0
        alpha_A = torch.pow(text_mask_A.clamp(0, 1), sharpness) * opacity_A
        
        # Ink Color with Internal Grain
        # Base color: Dark Brown to Black (Not just pure black)
        # R: 0.05-0.20 (Can be brownish)
        base_r = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.15 + 0.05
        base_g = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.10 + 0.05
        base_b = torch.rand(batch_size, 1, 1, 1, device=self.device) * 0.05 + 0.05
        base_ink_A = torch.cat([base_r, base_g, base_b], dim=1)
        
        # Add texture/grain to the ink color itself (so it's not a flat fill)
        ink_grain = torch.randn(batch_size, 3, H, W, device=self.device) * 0.05
        ink_color_A = (base_ink_A + ink_grain).clamp(0, 1)
        #ink_color_A = torch.zeros(batch_size, 3, 1, 1, device=self.device) # absolute black for binarized images
 
        
        # --- PREPARE BACKGROUND ---
        # Use the Stained Background from previous step
        # bg_rgb = self.create_stained_background_torch(batch_size, H, W) #synthetic
        bg_rgb = self.get_real_background(batch_size, H, W) #real patches from verona palimps
        # --- COMPOSITE (Subtractive Mixing) ---
        # --- TEXTURE INTEGRATION (The "Blend If" simulation) ---
        # Calculate luminance of the background (0 to 1)
        #bg_lum = bg_rgb.mean(dim=1, keepdim=True)
        
        # Modulate the alpha of the text based on the paper texture.
        # Ink soaks into dark valleys (low lum) and skips bright peaks (high lum).
        # We reduce alpha slightly where paper is very bright.
        #texture_mod = 1.0 - (bg_lum * 0.5) # Reduces alpha by up to 50% on bright spots
        
        #alpha_A = alpha_A * texture_mod
        #alpha_B = alpha_B * texture_mod

        #occlusion_mask = torch.clamp(alpha_A * 3.5, 0.0, 1.0)
        #alpha_B_occluded = alpha_B * (1.0 - occlusion_mask)
        #burn_factor_B = 1.0 - (alpha_B_occluded * (1.0 - ink_color_B))
        #canvas_with_B = bg_rgb * burn_factor_B

        #out = canvas_with_B * (1.0 - alpha_A) + ink_color_A * alpha_A
        # 1. Apply Undertext
        # Blend background towards ink_B
        out = bg_rgb * (1 - alpha_B) + ink_color_B * alpha_B
        #out = (1 - alpha_B) + ink_color_B * alpha_B # for binarized images
        # 2. Apply Uppertext
        # Blend result towards ink_A
        out = out * (1 - alpha_A) + ink_color_A * alpha_A

        # Apply Uneven Lighting (Optional but recommended)
        # out = self.apply_lighting_torch(out)

        # Map back to [-1, 1]
        #making overlapped chars on white background (for binarization)
        #white_bg = torch.ones_like(bg_rgb)
        #canvas_text_B = white_bg * burn_factor_B
        #gt_text = canvas_text_B * (1.0 - alpha_A) + ink_color_A * alpha_A

        # Create Background Target (Target B)
        #gt_bg = bg_rgb

        # Map back to [-1, 1]
        #return (out * 2.0) - 1.0, (gt_text * 2.0) - 1.0, (gt_bg * 2.0) - 1.0
        return (out * 2.0) - 1.0
        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']
        
    def gaussian_blur(self, img, kernel_size=5, sigma=1.0):
        """Applies Gaussian blur to a tensor image of shape (C, H, W)."""
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss = torch.exp(-(x**2)/(2*sigma**2))
        gauss = gauss / gauss.sum()
    
        # Apply separable convolution
        gauss = gauss.view(1, 1, -1)  # shape (1,1,K)
        img = img.unsqueeze(0)  # add batch dim -> (1,C,H,W)
    
        # Blur horizontally
        img = F.conv2d(img, weight=gauss.expand(img.size(1),1,-1), padding=(0,kernel_size//2), groups=img.size(1))
        # Blur vertically
        img = F.conv2d(img, weight=gauss.transpose(2,3).expand(img.size(1),1,-1), padding=(kernel_size//2,0), groups=img.size(1))
    
        return img.squeeze(0)

    def overlay_images(self, img1, img2, alpha=0.5, blur_sigma=0):
        """
        img1: tensor (C,H,W), top image
        img2: tensor (C,H,W), bottom image
        alpha: opacity of the bottom image (img2)
        blur_sigma: standard deviation for Gaussian blur
        """
        if blur_sigma > 0:
            img2 = self.gaussian_blur(img2, kernel_size=5, sigma=blur_sigma)
    
        # Overlay: img1 on top of img2
        combined = alpha * img2 + (1-alpha) * img1
        combined = torch.clamp(combined, 0, 1)
        return combined
    
    def forward(self):
        """
        Run forward pass; called by both functions <optimize_parameters>.
        We have another version of forward (forward_test) used in testing.
        """
        # --- FORCE MIXING START ---
        # Instead of random probability, explicitly set both labels to 1.
        # This guarantees that 'label_sum' is always 2.
        self.label[0] = 1
        self.label[1] = 1
        label_sum = torch.sum(self.label)
        # --- FORCE MIXING END ---
        
        if int(label_sum.item()) == 2:
            # both domains selected -> do palimpsest mixing
            self.real_input= self.palimpsest_mix(self.real_A, self.real_B)
            #self.real_input, self.real_A, self.real_B = self.palimpsest_mix(self.real_A, self.real_B) #for background removal
            #print("palimpsests mix in use")
        else:
            # fallback to original averaging behavior when only one domain is present
            self.real_input = (self.real_A * float(self.label[0].item()) + self.real_B*float(self.label[1].item())) / float(label_sum.item())

        #self.real_input = (self.real_A * self.label[0] + self.real_B * self.label[1]) / label_sum
        self.fake_all = self.netE(self.real_input)
        self.fake_A = self.netH1(self.fake_all)
        self.fake_B = self.netH2(self.fake_all)
        self.loss_sum = label_sum


    def compute_D_loss(self):
        """Calculate GAN loss and BCE loss for the discriminator"""
        batch_size = self.real_input.size(0)
        fake1 = self.fake_A.detach()
        fake2 = self.fake_B.detach()
        pred_fake1 = self.netD(0,fake1)
        pred_fake2 = self.netD(0,fake2)
        self.loss_D_fake = self.criterionGAN(pred_fake1, False) * self.label[0] \
                           + self.criterionGAN(pred_fake2, False) * self.label[1]

        # Real
        self.pred_real1 = self.netD(0,self.real_A)
        self.pred_real2 = self.netD(0,self.real_B)

        self.loss_D_real = self.criterionGAN(self.pred_real1, True) * self.label[0] \
                           + self.criterionGAN(self.pred_real2,True) * self.label[1]

        # BCE loss, netD(1) for the source prediction branch.
        self.predict_label = self.netD(1,self.real_input).view(batch_size, self.opt.max_domain)
        label_batch = self.label.unsqueeze(0).expand(batch_size, self.opt.max_domain)
        self.loss_BCE = self.criterionBCE(self.predict_label, label_batch)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_GAN # + self.loss_BCE * self.opt.lambda_BCE
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN loss, Ln loss, VGG loss for the generator"""
        # netD(0) for the separation branch.
        pred_fake1 = self.netD(0,self.fake_A)
        pred_fake2 = self.netD(0,self.fake_B)

        self.loss_G_GAN = self.criterionGAN(pred_fake1, True) * self.label[0] \
                          + self.criterionGAN(pred_fake2, True) * self.label[1]

        self.loss_Ln = self.criterionL1(self.real_A, self.fake_A) * 1.0 * self.label[0] \
                       + self.criterionL1(self.real_B,self.fake_B) * 30.0 * self.label[1]

        self.loss_VGG = self.criterionVGG(self.fake_A, self.real_A) * self.label[0] \
                        + self.criterionVGG(self.fake_B, self.real_B) * self.label[1]

        if self.criterionConn.alpha > 0:
            loss_conn_A = self.criterionConn(self.fake_A, self.real_A) * self.label[0]
            loss_conn_B = self.criterionConn(self.fake_B, self.real_B) * self.label[1]
            self.loss_Conn = loss_conn_B #+ loss_conn_B
        else:
            self.loss_Conn = 0.0

        # Feature consistency loss
        features_noisy = self.fake_all
        with torch.no_grad():
            features_clean = self.netE(self.real_B)
        loss_feature_consistency = self.criterionL1(features_noisy, features_clean)
        
        self.loss_G = self.loss_G_GAN * self.opt.lambda_GAN + self.loss_Ln * self.opt.lambda_Ln \
                      + self.loss_VGG * self.opt.lambda_VGG + self.loss_Conn * 1.0 + loss_feature_consistency *10.0

        return self.loss_G

    def forward_test(self):
        if self.opt.pre_mixed:
            self.real_input = self.real_A
            self.fake_all = self.netE(self.real_input)
            self.fake_A = self.netH1(self.fake_all)
            self.fake_B = self.netH2(self.fake_all)
            return
        label_sum = torch.sum(self.label)
        # Test case 1, test a single case only, write the output images.
        if self.opt.test_choice == 1:
            gt_label = [0] * self.opt.max_domain
            if 'A' in self.opt.test_input:
                gt_label[0] = 1
            if 'B' in self.opt.test_input:
                gt_label[1] = 1
            # if both components are present, use palimpsest-like mixing
            if int(label_sum.item()) == 2:
            # both domains selected -> do palimpsest mixing
                #self.real_input = self.overlay_images(self.real_A, self.real_B, alpha=0.5, blur_sigma=2.0)
                self.real_input = self.palimpsest_mix(self.real_A, self.real_B)
                print("palimpsests mix in use")
                print("real_input stats:", self.real_input.min(), self.real_input.max())
            else:
            # fallback to original averaging behavior when only one domain is present
                self.real_input = (self.real_A * float(self.label[0].item()) + self.real_B*float(self.label[1].item())) / float(label_sum.item())

            #self.real_input = (self.real_A * 1 + self.real_B * 1)
            self.fake_all = self.netE(self.real_input)
            self.predict_label = self.netD(1, self.real_input).view(self.opt.max_domain)
            predict_label = torch.where(self.predict_label > 0.0, 1, 0)
            self.fake_A = self.netH1(self.fake_all)
            self.fake_B = self.netH2(self.fake_all)
            if predict_label.tolist() == gt_label:
                self.correct = self.correct + 1
            self.all_count = self.all_count + 1
            if self.all_count == 300:
                print(self.correct / self.all_count)
        else:
            # Test case 0, test all cases, do not write output images.
            gt_label = [0] * self.opt.max_domain
            if 'A' in self.opt.test_input:
                gt_label[0] = 1
            if 'B' in self.opt.test_input:
                gt_label[1] = 1
            self.real_input = (self.real_A * gt_label[0] + self.real_B * gt_label[1])/ sum(gt_label)
            self.predict_label = self.netD(1, self.real_input).view(self.opt.max_domain)
            predict_label = torch.where(self.predict_label > 0.0, 1, 0)
            self.fake_all = self.netE(self.real_input)
            self.fake_A = self.netH1(self.fake_all)
            self.fake_B = self.netH2(self.fake_all)

            # Normalize to 0-1 for PSNR calculation.
            self.fake_A = (self.fake_A + 1)/2
            self.real_A = (self.real_A + 1)/2
            self.fake_B = (self.fake_B + 1)/2
            self.real_B = (self.real_B + 1)/2

            if gt_label[0] == 1:
                mse = self.criterionL2(self.fake_A, self.real_A)
                psnr = 10 * log10(1 / mse.item())
                self.psnr_count[0] += psnr
            if gt_label[1] == 1:
                mse = self.criterionL2(self.fake_B, self.real_B)
                psnr = 10 * log10(1 / mse.item())
                self.psnr_count[1] += psnr
            if predict_label.tolist() == gt_label:
                self.correct = self.correct + 1
            self.all_count = self.all_count + 1
            if self.all_count % 300 == 0:
                acc = self.correct / self.all_count
                print("Accuracy for current: ",acc)
                if gt_label[0] == 1:
                    print("PSNR_A: ", self.psnr_count[0]/ self.all_count)
                if gt_label[1] == 1:
                    print("PSNR_B: ", self.psnr_count[1]/ self.all_count)
                self.all_count = 0
                self.correct = 0
                self.psnr_count = [0] * self.opt.max_domain
                self.acc_all += acc
                self.test_time +=1
                if mean(gt_label) == 1:
                    print("Overall Accuracy:", self.acc_all/self.test_time)




