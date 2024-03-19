""" Evaluation script to evaluate Out-of-Context Detection Accuracy"""

import cv2
import os
import random
import torch
import time
import csv
from PIL import Image

current_directory = os.path.dirname(os.path.abspath(__file__))

from utils.config import *
from model_archs.models import CombinedModelMaskRCNN
from utils.common_utils import read_json_data
from utils.eval_utils import is_bbox_overlap, top_bbox_from_scores
from gill import models
from llama_cpp import Llama

from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

context_dict = {}

contextual_model = SentenceTransformer('sentence-transformers/stsb-bert-base')
model_name = 'img_use_rcnn_margin_10boxes_jitter_rotate_aug_ner'
combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
llm = Llama(model_path = current_directory + "/LLM/models/mistral-7b-openorca.Q4_0.gguf",
            n_gpu_layers=35, n_ctx=4096, use_gpu=True)
gill_model_dir = current_directory + '/gill/checkpoints/gill_opt'
gill_model = models.load_gill(gill_model_dir)

def get_scores(v_data):
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """
    checkpoint = torch.load(current_directory + '/models_final/' + model_name + '.pt')
    combined_model.load_state_dict(checkpoint)
    combined_model.to(device)
    combined_model.eval()

    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])
    bbox_list = v_data['maskrcnn_bboxes']
    bbox_classes = [-1] * len(bbox_list)
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    bbox_list.append([0, 0, img_shape[1], img_shape[0]])  # For entire image (global context)
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
  """, max_tokens= 100,  stop=["</pr>"], stream=True)

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
  return_outputs = gill_model.generate_for_images_and_texts(prompts, num_words=16, min_word_tokens=16)
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
    bbox_overlap = is_bbox_overlap(top_bbox_c1, top_bbox_c2, iou_overlap_threshold)

    embeddings_img_gill_cap = contextual_model.encode(gill_cap, convert_to_tensor=True)

    embeddings1 = contextual_model.encode(process_embedding1, convert_to_tensor=True)
    embeddings2 = contextual_model.encode(process_embedding2, convert_to_tensor=True)

    cosine_scores1_gill = util.cos_sim(embeddings1, embeddings_img_gill_cap)
    cosine_scores2_gill = util.cos_sim(embeddings2, embeddings_img_gill_cap)
    emds_sim = util.cos_sim(embeddings1, embeddings2)

    IC_NER_GILL = ((cosine_scores1_gill > 0.5 and len(v_data['caption1_entities']) < 1) \
                or (cosine_scores2_gill > 0.5 and len(v_data['caption2_entities']) < 1))


    if bbox_overlap:
        if textual_sim >= 0.5:
            cosmos_context = 0
            our_context = 0
        else:
            cosmos_context = 1

            llm_ans = QA_LLM(process_embedding1, process_embedding2)
            agree_ooc =  llm_ans[0] + llm_ans[1] + llm_ans[2] + llm_ans[3]
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
    return  cosmos_context, our_context

if __name__ == "__main__":
    """ Main function to compute out-of-context detection accuracy"""

    test_samples = read_json_data(os.path.join(DATA_DIR, 'test.json'))

    cosmos_correct = 0
    ours_correct = 0
    lang_correct = 0

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    total_time = 0

    if not os.path.isdir('output'): os.makedirs("output")

    with open('output/log.csv', 'w') as f:
        for i, v_data in tqdm(enumerate(test_samples)):
            actual_context = int(v_data['context_label'])
            language_context = 0 if float(v_data['bert_base_score']) >= textual_sim_threshold else 1
            pred_context_cosmos, pred_context_ours = evaluate_context_with_bbox_overlap(v_data)
            start_time = time.time()

            if pred_context_ours == actual_context:
                ours_correct += 1
            if pred_context_cosmos == actual_context:
                cosmos_correct += 1

            if language_context == actual_context:
                lang_correct += 1

            if actual_context == 1:
                if pred_context_ours == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if pred_context_ours == 0:
                    true_negative += 1
                else:
                    false_negative += 1

            end_time = time.time()

            time_run = start_time - end_time

            print(f"Image: {v_data['img_local_path']}")
            print(f"Actual Context Label: {actual_context}")
            print(f"Predicted Context Label (Ours): {pred_context_ours}")
            print(f"Inference Time (Ours): {time_run}")
            # print(f"Predicted Context Label (Cosmos): {pred_context_cosmos}")
            print("-------------------")

            # with open('output.txt', 'a') as f:
            f.write('img_path, actual_context, our_predicted_context, inference_time\n')
            f.write(f'{v_data['img_local_path']}, {actual_context}, {pred_context_ours}, {time_run}\n')
            # f.write(f'Predicted Time: {time_run}\n')
            # f.write('\n')

            total_time += time_run

        # print("Accuracy COSMOS", cosmos_correct / len(test_samples))
        accuracy_ours = ours_correct / len(test_samples)
        print("Accuracy OURS:", accuracy_ours)

        p_ooc = true_positive / (true_positive + false_positive)
        p_nooc = true_negative / (true_negative + false_negative)
        average_precision = (p_ooc + p_nooc) / 2
        print("Average Precision OURS:", average_precision)

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = (2 * precision * recall) / (precision + recall)
        print("F1-Score OURS:", f1_score)

    with open('output.txt', 'a') as f:
        f.write(f'Accuracy OURS: {accuracy_ours}\n')
        f.write(f'Average Precision OURS: {average_precision}\n')
        f.write(f'F1-Score OURS: {f1_score}\n')
        f.write(f'Total Time OURS: {total_time}\n')
