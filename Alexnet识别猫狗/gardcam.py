from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
model = resnet50(pretrained=True)
target_layer = model.layer4[-1]
# 设置输入
img = read_image(r"D:\Desktop\Practice\Alexnet识别猫狗\data\train\cat.0.jpg")
# 为选择的模型进行预处理
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# input_tensor = # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!
input_tensor=input_tensor.unsqueeze(0)
# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = 281

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(img, grayscale_cam)