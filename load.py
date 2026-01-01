from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_image_into_patches_pil(image: Image.Image):
    """
    Splits a PIL image into five patches: the full image and its four quadrants.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        List[PIL.Image.Image]: A list containing five PIL images corresponding to the full image,
                               top-left, top-right, bottom-left, and bottom-right patches.
    """
    # Ensure the image is in RGB mode
    image = image.convert('RGB')

    # Get image dimensions
    width, height = image.size
    mid_width, mid_height = width // 2, height // 2

    # Define the coordinates for the four quadrants
    boxes = [
        (0, 0, mid_width, mid_height),               # Top-left
        (mid_width, 0, width, mid_height),           # Top-right
        (0, mid_height, mid_width, height),          # Bottom-left
        (mid_width, mid_height, width, height)       # Bottom-right
    ]

    # Crop each quadrant
    patch_images = [image.crop(box) for box in boxes]

    # Combine the full image with the quadrant images
    return [image] + patch_images

end = '<|im_end|>\n'
ans = "<|im_start|>assistant\n The image is taken at"
human = "<|im_start|>Helix\n"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
end = '<|im_end|>\n'
ans = "<|im_start|>assistant\nAnswer:"
human = "<|im_start|>Helix\n"


def default_wrapping(input_text, embed_tokens, tokenizer, image_embeddings_list=None):
    """Wrap input text with system prompt and labeled image embeddings."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if image_embeddings_list is not None:
        system_prompt = (
            "<|im_start|>system\nYou are a helpful AI assistant named Snizel, trained by Hugging Face<|im_end|>"
            "\n<|im_start|>user\nUse the image data to answer the question below and don't answer anything unnecessary.\n"
        )
        input_text = '\n' + input_text + end + ans
    else:
        system_prompt = (
            "<|im_start|>system\nYou are a helpful human friend named Helix. You are friendly and helpful.\n"
            "Nikhil is your friend and a student while you are an assistant.\nComplete your turn of conversation below.\n"
            "<|im_end|>\n<|im_start|>user\n"
        )
        input_text = input_text + end + human

    # Tokenize and embed the system prompt
    system_prompt_ids = tokenizer(system_prompt, return_tensors='pt').input_ids.to(device)
    system_embeds = embed_tokens(system_prompt_ids).squeeze(0)  # Shape: [L_sys, D]

    # Initialize a list to hold all embeddings
    all_embeds = [system_embeds]

    if image_embeddings_list is not None:
        # Define descriptive labels for each image embedding
        image_labels = [
            "<img[global]>",      # Global image
            "<img[top-left]>",    # Top-left patch
            "<img[top-right]>",   # Top-right patch
            "<img[bottom-left]>", # Bottom-left patch
            "<img[bottom-right]>" # Bottom-right patch
        ]

        for image_embedding, label in zip(image_embeddings_list, image_labels):
            # Tokenize and embed the label
            label_ids = tokenizer(label, return_tensors='pt').input_ids.to(device)
            label_embeds = embed_tokens(label_ids).squeeze(0)  # Shape: [L_label, D]

            # Concatenate label embedding with the image embedding
            combined_image_embed = torch.cat([label_embeds, image_embedding.squeeze(0)], dim=0)  # Shape: [L_label + L_img, D]
            all_embeds.append(combined_image_embed)
        

    # Tokenize and embed the input text
    input_text_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    input_embeds = embed_tokens(input_text_ids).squeeze(0)  # Shape: [L_input, D]
    all_embeds.append(input_embeds)

    # Concatenate all embeddings along the sequence length dimension
    final_embeds = torch.cat(all_embeds, dim=0).unsqueeze(0)  # Shape: [L_total, D]
    print("Final embedding shape:", final_embeds.shape) # [1, L_total, D]

    return final_embeds


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
    top_k = min(top_k, logits.size(-1))  # Safety

    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        top_k_values, _ = torch.topk(logits, top_k)
        min_top_k = top_k_values[..., -1, None]
        logits = torch.where(logits < min_top_k, filter_value, logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_mask = cumulative_probs > top_p
        # Shift right to keep first token above threshold
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

def Auto_generator(
    input_embeddings,
    model,
    tokenizer,
    max_new_tokens=50,
    repetition_penalty=1.2,
    sentence_repetition_penalty=1.5,
    temperature=1.2,
    top_k=0,
    top_p=1
):
    model.eval()

    device = input_embeddings.device
    dtype = input_embeddings.dtype
    past_input = input_embeddings
    
    # Initialize generation
    generated_ids = []
    sentence_history = []
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Process with regular input (no caching for now)
            outputs = model(inputs_embeds=past_input)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]

            # Repetition penalty (token-level)
            for token_id in set(generated_ids):
                logits[0, token_id] /= repetition_penalty

            # Decode current sequence for sentence history
            current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            current_sentences = [s.strip().lower() for s in current_text.split('.') if s.strip()]
            last_sentence = current_sentences[-1] if current_sentences else ""

            # Sentence-level repetition penalty
            for sentence in sentence_history:
                if last_sentence and sentence.startswith(last_sentence):
                    likely_next_ids = tokenizer.encode(sentence[len(last_sentence):], add_special_tokens=False)
                    for tok_id in likely_next_ids:
                        if tok_id < logits.shape[-1]:
                            logits[0, tok_id] /= sentence_repetition_penalty

            # Apply temperature
            logits = logits / temperature

            # Apply top-k and top-p sampling
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_token = next_token_id.item()
            
            if generated_token == tokenizer.eos_token_id:
                break

            generated_ids.append(generated_token)

            # Update sentence history
            updated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            new_sentences = [s.strip().lower() for s in updated_text.split('.') if s.strip()]
            sentence_history = list(set(sentence_history + new_sentences))

            # Append next token embedding
            next_token_embed = model.get_input_embeddings()(next_token_id).to(dtype)
            past_input = torch.cat([past_input, next_token_embed], dim=1)
            
    return generated_ids

class SimpleMLP(nn.Module):
    def __init__(self, c, text_dim, hidden_dim=1024):
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c * text_dim)
        )
        self.c = c
        self.text_dim = text_dim

    def forward(self, x):
        # x: (b, 1, 512)
        x = x.squeeze(1)  # (b, 512)
        x = self.mlp(x)   # (b, c * text_dim)
        x = x.view(-1, self.c, self.text_dim)  # (b, c, text_dim)
        return x

    def generate(self, 
            main_model, 
            main_tokenizer, 
            image_model, 
            image_processor,
            Qmodel, 
            input_text, 
            image = None, 
            wrapping_function = None, 
            max_new_tokens=50, 
            Repetation_penalty=1.2,
            sentence_repeat_penalty=1.5,
            temperature=1.2,
            top_k=50,
            top_p=0.95 
            ):
            device = main_model.device
            image_embeddings = None
            if image is not None:
                # Get the current device from the model
                inp_tok = main_tokenizer(input_text, return_tensors="pt").input_ids.to(device)  # [1, q_len]
                Question_embeddings = Qmodel(inp_tok).to(torch.float32) # [1, 1, 576]
                
                # Process image and move to correct device
                image_inputs = image_processor(images=image, return_tensors="pt")
                image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                image_embeddings = image_model.get_image_features(image_inputs['pixel_values']).unsqueeze(1)  # [1, 1, 512]
                image_embeddings = self.forward(image_embeddings).to(main_model.dtype)  # [1, 64, 576]

            if wrapping_function is not None:
                input_embeddings = wrapping_function(input_text, main_model.model.embed_tokens, main_tokenizer, image_embeddings)  # [1, x, 576]
            else:
                input_embeddings = default_wrapping(input_text, main_model.model.embed_tokens, main_tokenizer, image_embeddings)  # [1, x, 576]
            
            # Enable use_cache for more efficient generation
            output_tokens = Auto_generator(
                input_embeddings, 
                main_model, 
                main_tokenizer, 
                max_new_tokens=max_new_tokens, 
                repetition_penalty=Repetation_penalty, 
                sentence_repetition_penalty=sentence_repeat_penalty, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p
            )
            
            return output_tokens

