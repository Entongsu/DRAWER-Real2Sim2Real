import os
import torch

# tokenizer to use
# TOKENIZER = "microsoft/Multilingual-MiniLM-L12-H384"
TOKENIZER = "openai/clip-vit-large-patch14"
# TOKENIZER = "openai/clip-vit-base-patch32"

# maximum number of words in a language goal
LANG_MAX_WORD_LEN = 25

# language embedding obs key name
LANG_OBS_KEY = "lang_embed"

# (HACK) enable language-vision multiplication, post spatial softmax layer
# LANG_VIS_MULT_ENABLED = True
LANG_VIS_MULT_ENABLED = False

# these global variables will be populated automatically

# whether language conditioning is enabled
LANG_COND_ENABLED = False

# these global variables will be populated lazily
LANG_EMB_MODEL = None
TZ = None

def init_lang_model():
    from transformers import AutoModel, pipeline, AutoTokenizer, CLIPTextModelWithProjection

    os.environ["TOKENIZERS_PARALLELISM"] = "true" # needed to suppress warning about potential deadlock
    global LANG_EMB_MODEL
    global TZ

    # CLIP
    LANG_EMB_MODEL = CLIPTextModelWithProjection.from_pretrained(
        TOKENIZER,
        cache_dir=os.path.expanduser("~/tmp/clip")
    ).eval()

    TZ = AutoTokenizer.from_pretrained(TOKENIZER, TOKENIZERS_PARALLELISM=True)
    
    # # MiniLM
    # # https://github.com/microsoft/unilm/tree/master/minilm
    # LANG_EMB_MODEL = AutoModel.from_pretrained(TOKENIZER, cache_dir=os.path.expanduser("~/tmp/minilm")).eval()

    # # pip install --no-cache-dir transformers sentencepiece
    # TZ = AutoTokenizer.from_pretrained(TOKENIZER, TOKENIZERS_PARALLELISM=True, use_fast=False)


def get_lang_emb(lang):
    if lang is None:
        return None

    if TZ is None:
        init_lang_model()
    
    num_words = len(lang.split())
    if num_words > LANG_MAX_WORD_LEN:
        raise Exception("Number of words {} in sentence {} exceeded max length {}".format(num_words, lang, LANG_MAX_WORD_LEN))

    with torch.no_grad():
        tokens = TZ(
            text=lang,                   # the sentence to be encoded
            add_special_tokens=True,             # Add [CLS] and [SEP]
            max_length=LANG_MAX_WORD_LEN,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,        # Generate the attention mask
            return_tensors="pt",               # ask the function to return PyTorch tensors
        )
        # lang_emb = LANG_EMB_MODEL(**tokens)["pooler_output"].detach()[0]
        lang_emb = LANG_EMB_MODEL(**tokens)['text_embeds'].detach()[0]
        lang_emb = lang_emb.cpu().numpy()

    return lang_emb