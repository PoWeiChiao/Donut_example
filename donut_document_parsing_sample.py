import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re

def main():
    # Load the processor and model
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", force_download=True)
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", force_download=True)

    # Move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Load a document image (replace with your own image path)
    image = Image.open("path").convert("RGB")

    # Preprocess the image
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,)
    
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    print(processor.token2json(sequence))

if __name__ == "__main__":
    main()