import os
from pathlib import Path
import json

import ollama
from ollama import Options

import gradio as gr
import torch
from PIL import Image

# models
from models.image_editing import ImageEditing
from models.bg_removing import BackgroundRemover
from models.bg_editing import BackgroundEditing
from models.image_generation import ImageGeneration

from deep_translator import GoogleTranslator

origins = [
     'https://localhost:3000',
     'http://localhost:3000',
     'localhost:3000',
     '*'
]

# environmental variables
OLLAMA_CONNECTION_STR = os.environ.get(
    "OLLAMA_CONNECTION_STR", "http://localhost:11434"
)  # local url
OLLAMA_MODEL = os.environ.get("MODEL_NAME", "qwen2.5:14b-instruct-q4_K_M")  # ollama model name

TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.3))  # temperature for json generation
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", 4096))  # input context length
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 512))  # max tokens for output
BASE_WIDTH = int(os.environ.get("BASE_WIDTH", 300))

MODEL_LIST_PATH = os.environ.get("MODEL_LIST_PATH", "prompts/model_list.json")
JSON_PROMPT_PATH = os.environ.get("JSON_PROMPT_PATH", "prompts/json_prompt.txt")
JSON_SCHEMA_PATH = os.environ.get("JSON_SCHEMA_PATH", "prompts/json_schema.txt")

HISTORY = []  # TO CHANGE !!!
IMG_FOLDER = 'images/'  # TO CHANGE !!!
IMG_CNT = len(os.listdir(IMG_FOLDER[:-1])) # TO CHANGE !!!

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_edit_model = ImageEditing()
img_gen_model = ImageGeneration()
bg_remover = BackgroundRemover()
bg_editor = BackgroundEditing()

img_edit_model.pipe.to(DEVICE)
img_gen_model.pipe.to(DEVICE)
bg_editor.pipe.to(DEVICE)

translator = GoogleTranslator(source='auto', target='en')

torch.cuda.empty_cache()

MODELS = {0: img_gen_model,
          1: img_edit_model,
          2: bg_remover,
          3: bg_editor,}

"""
img_gen_model = ImageGeneration()
img_gen_model.pipe.to("cpu")

torch.cuda.empty_cache()
"""

def read_file(input_path: str) -> str:
    file_path = Path(input_path)
    with file_path.open('r', encoding='utf-8') as f:
        text = f.read()
    return text


def generate_json(payload: dict) -> dict:
    """
    Generates .json version of summary

    Args:
      request (Request): {"user_message": text to summarize, 
                          "img": img_id,
                          "history": history}

    Returns:
      dict: {"model_id": model id,
            "prompt": prompt,
            "image_type": 'no', 'last', or 'previous'}
    """
    # read arguments
    user_message = payload["user_message"]
    user_img_id = payload["user_img"]
    
    #history = payload.get("history")  CHANGE LATER !!!
    
    prompt = read_file(JSON_PROMPT_PATH)
    json_schema = read_file(JSON_SCHEMA_PATH)
    model_list = read_file(MODEL_LIST_PATH)
    
    hst = prepare_history(HISTORY)
    
    prompt = prompt.replace("$USER_MSG", user_message)  # add user message
    prompt = prompt.replace("$SCHEMA", json_schema)  # add schema
    prompt = prompt.replace("$MODEL_LIST", model_list)  # add model list
    prompt = prompt.replace("$HISTORY", hst)  # add history  CHANGE LATER !!!
    
    api_response = ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    format='json',
                    stream=False,
                    options=Options(
                        num_ctx=CONTEXT_LENGTH,
                        num_predict=-1,
                        temperature=TEMPERATURE,
        ),
    )
    response = api_response["response"]
    
    update_history_user(user_message, user_img_id)
    update_history_model(json.loads(response))
    
    # logs
    print(hst, end='\n\n')
    return response


def save_image(img: Image):
    num_files = len(os.listdir(IMG_FOLDER))
    img.save(IMG_FOLDER + str(num_files) + '.png')
    

def prepare_history(hst):
    hst4model = [a['user_message'] for a in hst if 'user_message' in a.keys()]
    return str(hst4model)


def get_chosen_model(llm_response):
    model_id = llm_response["model_id"]
    with open(MODEL_LIST_PATH, 'r', encoding='utf-8') as f:
        models = json.load(f)['models']   
    chosen_model = None
    
    for m in models:
        if m["id"] == model_id:
            chosen_model = m
    
    if chosen_model is None:
        raise Exception(404, "LLM refers to unknown model")
    return chosen_model
            

def update_history_user(user_msg="-", image_id="-"):
    global IMG_CNT
    if user_msg is None:
        user_msg = "-"
    if image_id is None:
        image_id = "-"
        
    if image_id != "-":
        IMG_CNT += 1
        
    HISTORY.append({
        "user_message": user_msg,
        "image_id": image_id,
    },) 
    

def update_history_model(llm_response):
    global IMG_CNT
    history_sample = {'model_message': "-",
                      'image_id': "-"}
    history_sample["image_id"] = IMG_CNT
    IMG_CNT += 1
    HISTORY.append(history_sample)


def process_input(text, image):
    global HISTORY
    if text == "\\reset":
        HISTORY = []
        return "History deleted", None
    
    if text == "\\rm images":
        return "TODO add function that clears img folder", None
    
    image_id = IMG_CNT if image else None
    llm_output = generate_json({"user_message": text,
                                "user_img": image_id})
    
    if image:
        save_image(image)
    
    json_output = json.loads(llm_output)
    img_out = None

    chosen_model = get_chosen_model(json_output)
    model_id = chosen_model["id"]
    
    bgr = None
    if model_id == 3:
        bgr = bg_remover
    
    img_out = model_handler(MODELS[model_id], json_output, image, translate=True, bg_remover=bgr)
    save_image(img_out)
    
    return llm_output, img_out


def resize(img: Image) -> Image:
    ratio = (BASE_WIDTH / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(ratio)))
    
    return img.resize((BASE_WIDTH, h_size), Image.Resampling.LANCZOS)


def model_handler(model, json_output, image=None, translate=True, bg_remover=None):
    img_type = json_output["image_type"]
    text = json_output["prompt"]
    if translate:
        query = translator.translate(text) 
        
    input_image = None
    
    if len(os.listdir(IMG_FOLDER[:-1])) == 0:
        input_image = image
    else:
        if img_type == "previous" and len(os.listdir(IMG_FOLDER[:-1])) >= 2:
            filename = os.listdir(IMG_FOLDER[:-1])[-2]
        else:
            filename = os.listdir(IMG_FOLDER[:-1])[-1]
        input_image = Image.open(IMG_FOLDER + filename)
        
    input_image = resize(input_image)    
        
    if bg_remover:
        img_out = model.predict({"text": query, "image": input_image}, bg_remover)
    else:
        img_out = model.predict({"text": query, "image": input_image}, bg_remover)
    torch.cuda.empty_cache()
    return img_out


iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(label="Текст (необязательно)"),
        gr.Image(type="pil", label="Изображение (необязательно)")
    ],
    outputs=[
        gr.Textbox(label="Обработанный текст"),
        gr.Image(type="pil", label="Обработанное изображение")
    ],
    title="Обработка Текста и Изображений",
    description="Загрузите текст, изображение или и то, и другое.  Результат будет показан ниже.",
    allow_flagging="never"
)

iface.launch(server_port=8899)
