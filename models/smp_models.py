import segmentation_models_pytorch as smp

class SMPUNet(smp.Unet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        x = x['image']
        x = super().forward(x)
        return {'logit': x}


class SMPDeepLab(smp.DeepLabV3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        x = x['image']
        x = super().forward(x)
        return {'logit': x}    


class SMPFCN(smp.FPN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        x = x['image']
        x = super().forward(x)
        return {'logit': x}    

class SMPPSPNet(smp.PSPNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        x = x['image']
        x = super().forward(x)
        return {'logit': x}   

