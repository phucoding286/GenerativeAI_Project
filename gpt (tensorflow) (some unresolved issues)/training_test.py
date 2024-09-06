from __wordTokenizer import WordTokenizer
from __GPT import GPT
from __createMask import create_masks
from __gptTrainingFunction import training
import keras as keras
import tensorflow as tf

with open("source_sequences.txt", 'r', encoding='utf8') as f: x_sample = f.read().splitlines()[:100]
with open("target_sequences.txt", 'r', encoding='utf8') as f: y_sample = f.read().splitlines()[:100]
x_sample += y_sample
y_sample += x_sample

tokenizer = WordTokenizer(x_sample, y_sample, maxlen_word=50)
mask = create_masks(tokenizer.x_batch_idx_word)

# model = keras.models.load_model("model.keras")
model = keras.Sequential([
    GPT(layer_num=4, num_heads=64, d_model=512, ffn_hidden=1024, dropout=0.001, vocab_size=len(tokenizer.word_to_idx)+1)
])
training(tokenizer.x_batch_idx_word, tokenizer.y_batch_idx_word, model, mask, batch_size=16, epochs=50, lr=0.001, step_verbose=True)

# test output
output = tf.argmax(model(tokenizer.x_batch_idx_word[:3], mask=mask[:3]), axis=-1)
print(tokenizer.output_model_decode(output))
# model.save("model.keras")

while True:
    user_input = tokenizer.word_encode([input("Bạn: ")])
    mask = create_masks(user_input)
    output = tf.argmax(model(user_input, mask=mask), -1)
    print(tokenizer.output_model_decode(output))
