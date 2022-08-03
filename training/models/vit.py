from vit_keras import vit, utils



def get_vit():
    image_size = 224
    return  vit.vit_b16(
    image_size=image_size,
    activation='sigmoid',
    pretrained=False,
    include_top=True,
    pretrained_top=False
)

