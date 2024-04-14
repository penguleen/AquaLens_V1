import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image
# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained="laion2b_s34b_b79k") #laion2b_s34b_b79k #ViT-B-32 #'ViT-H-14', pretrained="laion2b_s34Vb_b79k" #ViT-B-16-plus-240, laion400m_e31
model.to(device)

# Takes an imgae and convert into RGB format, preprocess then encodes it using openclip model
def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

# Compares 2 images using OpenCV and encodes them using imageEncoder function to produce similarity result.
def generateScore(image1, image2):
    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)       # Compute using cosine similarity between encoder images 1 and 2.
    score = round(float(cos_scores[0][0])*100, 2)       # Returns a rounded similarity score as a percentage
    return score

def collection(ideal_imagefiles, imagefiles):
    print(f"similarity Score: ", round(generateScore("./video_frames/fish_000006.jpg", "./5classes/steve/steve_3058.jpg"), 2))

