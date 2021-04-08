from src.multilingual_clip import MultilingualClip

model_path = 'M-CLIP/Swedish-500k'
tok_path = 'M-CLIP/Swedish-500k'
weight_name='Swedish-500k-Linear-Weights.pkl'

sweclip_args = {'model_name': model_path,
                'tokenizer_name': tok_path,
                'head_name': weight_name,}

sweclip = MultilingualClip(**sweclip_args)

print(sweclip('test'))