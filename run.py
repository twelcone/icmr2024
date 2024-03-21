""" Evaluation script to evaluate Out-of-Context Detection Accuracy"""

from tqdm import tqdm
from utils.eval_utils import is_bbox_overlap, top_bbox_from_scores
from utils.common_utils import read_json_data
from model_archs.models import CombinedModelMaskRCNN
from utils.config import *
import cv2
import os
import sys
import random
import time
import torch
import csv
from PIL import Image

current_directory = os.path.dirname(os.path.abspath(__file__))

from gill import models
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util

if not os.path.exists(os.path.join(OUTPUT_DIR, "log_model_info")):
    os.makedirs(os.path.join(OUTPUT_DIR, "log_model_info"))
if not os.path.exists(os.path.join(OUTPUT_DIR, "results")):
    os.makedirs(os.path.join(OUTPUT_DIR, "results"))

def get_model_info(model, f) -> None:
    params = 0
    size_model = 0

    for param in model.parameters():
        params += param.numel()
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    f.write(f"model params: {params}\n")
    f.write(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

contextual_model = SentenceTransformer('sentence-transformers/stsb-bert-base')

model_name = 'img_use_rcnn_margin_10boxes_jitter_rotate_aug_ner'
combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
checkpoint = torch.load(
    current_directory + '/models_final/' + model_name + '.pt')
combined_model.load_state_dict(checkpoint)
combined_model.to(device)
combined_model.eval()

tmp = sys.stderr

sys.stderr = open(os.path.join(OUTPUT_DIR, "log_model_info/llm_model.txt"), 'w')
llm = Llama(model_path=current_directory + "/LLM/models/mistral-7b-openorca.Q4_0.gguf",
            n_gpu_layers=35, n_ctx=4096, use_gpu=True)
sys.stderr = tmp

gill_model_dir = current_directory + '/gill/checkpoints/gill_opt'
gill_model = models.load_gill(gill_model_dir)

### Get model params and storage
with open(os.path.join(OUTPUT_DIR, "log_model_info/contextual_model.txt"), "w") as f:
    get_model_info(contextual_model, f)
with open(os.path.join(OUTPUT_DIR, "log_model_info/gill_model.txt"), "w") as f:
    get_model_info(gill_model, f)
with open(os.path.join(OUTPUT_DIR, "log_model_info/cosmos_model.txt"), "w") as f:
    get_model_info(combined_model, f)


def get_scores(v_data):
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """

    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])
    bbox_list = v_data['maskrcnn_bboxes']
    bbox_classes = [-1] * len(bbox_list)
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    # For entire image (global context)
    bbox_list.append([0, 0, img_shape[1], img_shape[0]])
    bbox_classes.append(-1)
    cap1 = v_data['caption1_modified']
    cap2 = v_data['caption2_modified']

    img_tensor = [torch.tensor(img).to(device)]
    bboxes = [torch.tensor(bbox_list).to(device)]
    bbox_classes = [torch.tensor(bbox_classes).to(device)]

    embed_c1 = torch.tensor(use_embed([cap1]).numpy()).to(device)
    embed_c2 = torch.tensor(use_embed([cap2]).numpy()).to(device)

    with torch.no_grad():
        z_img, z_t_c1, z_t_c2 = combined_model(img_tensor, embed_c1, embed_c2, 1, [embed_c1.shape[1]],
                                               [embed_c2.shape[1]], bboxes, bbox_classes)

    z_img = z_img.permute(1, 0, 2)
    z_text_c1 = z_t_c1.unsqueeze(2)
    z_text_c2 = z_t_c2.unsqueeze(2)

    # Compute Scores
    score_c1 = torch.bmm(z_img, z_text_c1).squeeze()
    score_c2 = torch.bmm(z_img, z_text_c2).squeeze()

    return score_c1, score_c2


def QA_LLM(cap1, cap2):
    output = llm.create_completion(f"""
    <pr> I will ask you six questions about two given sentences. Rate your answers on a scale from 0 to 9.

    1. Are the sentences out of context?
    2. Are the subject matters different?
    3. Is the broader context different?
    4. Do the sentences cohere together?
    5. Do the sentences exhibit contextual consistency?
    6. Determine the semantic similarity.

    Sentences: [caption 1, caption 2]. You should output a Python list of length 6 (each component is a rate value) only without explanations.
    <caption 1> {cap1}
    <caption 2> {cap2}

    </pr>
    """, max_tokens=100,  stop=["</pr>"], stream=True)

    cnt = 0
    ans = []
    lmt = 6
    for token in output:
        if token["choices"][0]["text"].isdigit() == True:
            cnt += 1
            ans.append(int(token["choices"][0]["text"]))
        if cnt == lmt:
            break
    while len(ans) != 6:
        ans.append(random.randint(0, 9))

    return ans


def generate_gill_caption(img_path: str) -> str:
    img = Image.open(img_path)
    img = img.convert('RGB')
    prompts = [
        img,
        'Caption of the image: '
    ]
    return_outputs = gill_model.generate_for_images_and_texts(
        prompts, num_words=16, min_word_tokens=16)
    return str(return_outputs[0]).strip()


def evaluate_context_with_bbox_overlap(v_data):
    """
        Computes predicted out-of-context label for the given data point

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
    """

    img_path = os.path.join(DATA_DIR, v_data['img_local_path'])
    gill_cap = generate_gill_caption(img_path)

    bboxes = v_data['maskrcnn_bboxes']
    score_c1, score_c2 = get_scores(v_data)
    textual_sim = float(v_data['bert_base_score'])

    process_embedding1 = v_data['caption1']
    process_embedding2 = v_data['caption2']

    top_bbox_c1 = top_bbox_from_scores(bboxes, score_c1)
    top_bbox_c2 = top_bbox_from_scores(bboxes, score_c2)
    bbox_overlap = is_bbox_overlap(
        top_bbox_c1, top_bbox_c2, iou_overlap_threshold)

    embeddings_img_gill_cap = contextual_model.encode(
        gill_cap, convert_to_tensor=True)

    embeddings1 = contextual_model.encode(
        process_embedding1, convert_to_tensor=True)
    embeddings2 = contextual_model.encode(
        process_embedding2, convert_to_tensor=True)

    cosine_scores1_gill = util.cos_sim(embeddings1, embeddings_img_gill_cap)
    cosine_scores2_gill = util.cos_sim(embeddings2, embeddings_img_gill_cap)
    emds_sim = util.cos_sim(embeddings1, embeddings2)

    IC_NER_GILL = ((cosine_scores1_gill > 0.5 and len(v_data['caption1_entities']) < 1)
                   or (cosine_scores2_gill > 0.5 and len(v_data['caption2_entities']) < 1))

    if bbox_overlap:
        if textual_sim >= 0.5:
            cosmos_context = 0
            our_context = 0
        else:
            cosmos_context = 1

            llm_ans = QA_LLM(process_embedding1, process_embedding2)
            agree_ooc = llm_ans[0] + llm_ans[1] + llm_ans[2] + llm_ans[3]
            disagree_ooc = llm_ans[4] + llm_ans[5]

            if agree_ooc >= 28.8 and disagree_ooc < 3.6:
                our_context = 1
            elif IC_NER_GILL:
                our_context = 0
            elif emds_sim >= textual_sim_threshold:
                our_context = 0
            else:
                our_context = 1
    else:
        return 0, 0
    return cosmos_context, our_context


if __name__ == "__main__":
    """ Main function to compute out-of-context detection accuracy"""

    test_samples = read_json_data(os.path.join(DATA_DIR, 'test.json'))
    ours_correct = 0

    log_file = open(os.path.join(OUTPUT_DIR, "results/log_result_our_methods.csv"), "w")
    csvwriter = csv.writer(log_file)
    fields = ["img_path", "actual_context", "predicted_context", "inference_time"]
    csvwriter.writerow(fields)

    for i, v_data in tqdm(enumerate(test_samples)):
        start_time = time.time()
        actual_context = int(v_data['context_label'])

        _, pred_context_ours = evaluate_context_with_bbox_overlap(
            v_data)
        end_time = time.time()
        if pred_context_ours == actual_context:
            ours_correct += 1

        csvwriter.writerow([
            v_data['img_local_path'],
            actual_context,
            pred_context_ours,
            end_time - start_time
        ])
        print(f"Image: {v_data['img_local_path']}")
        print(f"Actual Context Label: {actual_context}")
        print(f"Predicted Context Label: {pred_context_ours}")
        print("-------------------")

    log_file.close()
    print("Accuracy:", ours_correct / len(test_samples))
