import efficientnet.keras as efn 

def get_efficient_b0():
    return efn.EfficientNetB0(weights=None)
