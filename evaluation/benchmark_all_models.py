from tqdm import tqdm
from constants import *
from torchvision.transforms.functional import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from modelvshuman.models.wrappers.pytorch import PytorchModel, PyContrastPytorchModel, ClipPytorchModel
import modelvshuman
import os
from modelvshuman import models
from captum.attr import *
from utils import *
from metrics import *
from explainability import *
import os
import clip as cliplib
import pandas as pd
torch.cuda.empty_cache()
zoo = modelvshuman.models.pytorch.model_zoo
BATCH_SIZE = 1
EXPLANATIONS_FOLDER = 'explanations/'
print('### PYTORCH MODELS ######')


def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def undo_default_preprocessing(images):
    """Convenience function: undo standard preprocessing."""
    assert type(images) is torch.Tensor
    default_mean = torch.Tensor([0.485, 0.456, 0.406]).to(device())
    default_std = torch.Tensor([0.229, 0.224, 0.225]).to(device())

    images *= default_std[None, :, None, None]
    images += default_mean[None, :, None, None]

    return images


def _clip_preprocessing(images):
    """Convenience function: undo standard preprocessing."""
    assert type(images) is torch.Tensor
    default_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(device())
    default_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(device())

    images = normalize(images, default_mean, default_std)

    return images


def _tf_to_torch(t):
    t = tf.cast(t, tf.float32).numpy()
    if t.shape[-1] in [1, 3]:
        t = np.moveaxis(t, -1, 1)

    t = torch.tensor(t, requires_grad=True).cuda()

    return t


def _torch_to_tf(t):
    try:
        t = t.detach()
    except:
        pass
    try:
        t = t.cpu()
    except:
        pass
    try:
        t = t.numpy()
    except:
        pass
    t = np.array(t)
    if t.shape[1] in [1, 3]:
        t = np.moveaxis(t, 1, -1)

    t = np.array(t, np.float32)

    return t


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = cliplib.tokenize(texts)  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def process_model(model_name, show_examples=False, clip=False):
    explanation_methods = [saliency, input_gradient, smoothgrad, integrad]
    if clip:
        #model_clip = getattr(zoo, model_name)(model_name)
        model, preprocess = cliplib.load(model_name)
        model = model.cpu()
        zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates, model)

        model.eval()
        print(zeroshot_weights)
    else:
        model = getattr(zoo, model_name)(model_name)
        zeroshot_weights = None
        model = model.model.cuda()
    RESULTS = []
    scores_acc = []
    img_passed = 0
    ds = get_dataset(BATCH_SIZE, True)

    REDUCERS = [1, 4, 16]

    for x_batch, h_batch, y_batch in ds:

        x_batch = _tf_to_torch(x_batch)
        x_batch = x_batch - x_batch.min()
        x_batch = x_batch/x_batch.max()
        y_batch = _tf_to_torch(y_batch)

        y_ohe = torch.argmax(y_batch, axis=-1)

        sa_phi = saliency(model, x_batch, y_batch, clip=clip, zsw=zeroshot_weights)
        sa_phi = torch.abs(torch.mean(sa_phi, axis=1))

        image_features = model.encode_image(x_batch)
        image_features = image_features/image_features.norm(dim=-1, keepdim=True)
        y_pred = 100.*image_features@zeroshot_weights

        s = torch.argmax(y_pred, -1) == y_ohe
        for _s in s:
            scores_acc.append(_s.cpu().numpy())
        print('mean score of the model:', np.mean(scores_acc))

        label = _torch_to_tf(y_batch)
        label = np.argmax(label, -1)
        #import pdb;pdb.set_trace()
        for phi_name, phi in [('saliency', sa_phi)]:

            dice_score = dice(_torch_to_tf(phi), h_batch, reducers=REDUCERS)
            spearman_score = spearmanr_sim(_torch_to_tf(phi), h_batch, reducers=REDUCERS)
            iou_score = iou(_torch_to_tf(phi), h_batch, reducers=REDUCERS)

            for metric_name, scores in [('dice', dice_score), ('spearman', spearman_score), ('iou', iou_score)]:
                for reducer_key in REDUCERS:
                    for img_i in range(len(scores[reducer_key])):
                        img_id = img_passed + img_i
                        RESULTS.append(
                            (model_name, phi_name, img_id, label[img_i], metric_name, reducer_key, scores[reducer_key][img_i].astype(np.float32)))

            phi = _torch_to_tf(phi)
            for pid, p in enumerate(phi):
                p -= p.min()
                p /= p.max()
                p *= 255.0

                img_id = pid + img_passed
                cv2.imwrite(
                    f"explanations/{model_name}_{phi_name}_{img_id}_{label[pid]}.png",  p.astype(np.uint8))

        img_passed += len(x_batch)
        print(model_name, img_passed)

    res = pd.DataFrame(RESULTS, columns=[
        'model name',
        'attribution name',
        'image_id',
        'label',
        'metric_name',
        'reducer',
        'metric_score'
    ])
    if clip:
        saving_name = model_name.replace('/', '')
        file_name = f'clip_{saving_name}_results.csv'
    else:
        file_name = f'{model_name}_results.csv'

    res.to_csv(file_name)


def benchmarks(list_models, show_examples=False):
    print('Creating output')
    os.makedirs(EXPLANATIONS_FOLDER, exist_ok=True)
    for model_name in list_models:  # models.list_models("pytorch")[61:]:
        print(f'\n /\ STarting Processing of model {model_name}')
        process_model(model_name, show_examples=False, clip=True)


def main():
    list_models = models.list_models("pytorch")
    print('Processing the following models:')

    list_models = ['ViT-L/14']
    benchmarks(list_models)


main()
