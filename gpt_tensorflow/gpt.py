import keras
import tensorflow as tf
import math
import numpy as np

class AutoRgressiveLayer(keras.layers.Layer):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout, eps=1e-5, **kwargs):
        super(AutoRgressiveLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dims = d_model // num_heads
        self.eps = eps
    

    def build(self, input_shape):
        # attention layers
        self.qkv_w = keras.layers.Dense(3 * self.d_model)
        self.qkv_out = keras.layers.Dense(self.d_model)
        self.dropout_attention_product_out = keras.layers.Dropout(self.dropout)

        # ffn layers
        self.ffn_w1 = keras.layers.Dense(self.ffn_hidden)
        self.ffn_hidden_dropout = keras.layers.Dropout(self.dropout)
        self.ffn_w2 = keras.layers.Dense(self.d_model, activation="relu")
        self.ffn_out_dropout = keras.layers.Dropout(self.dropout)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=self.eps)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=self.eps)
        return super(AutoRgressiveLayer, self).build(input_shape)
    
    # self attention block
    def att_block(self, x: tf.Tensor, mask: tf.Tensor):
        batch_size, max_sequence_length, input_dims = x.shape
        qkv = self.qkv_w(x)
        qkv = tf.reshape(qkv, shape=(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dims))
        qkv = tf.transpose(qkv, perm=(0, 2, 1, 3))
        q, k, v, = tf.split(qkv, num_or_size_splits=3, axis=-1)
        d_k = q.shape[-1]
        k_t = tf.transpose(k, perm=(0, 1, 3, 2))
        scaled = tf.matmul(q, k_t) / math.sqrt(d_k)

        if mask is not None:
            scaled = tf.transpose(scaled, perm=(1, 0, 2, 3)) + mask
            scaled = tf.transpose(scaled, perm=(1, 0, 2, 3))

        attention = tf.nn.softmax(scaled, axis=-1)
        values = tf.matmul(attention, v)
        values = tf.transpose(values, perm=(0, 2, 1, 3))
        values = tf.reshape(values, shape=(batch_size, max_sequence_length, self.d_model))
        attention_out = self.qkv_out(values)
        return self.dropout_attention_product_out(attention_out)
    
    # feedforward block
    def ffn_block(self, x):
        x = self.ffn_w1(x)
        x = self.ffn_hidden_dropout(x)
        x = self.ffn_w2(x)
        x = self.ffn_out_dropout(x)
        return x


    def call(self, x: tf.Tensor, causal_mask: tf.Tensor):
        pre_x = tf.identity(x)
        x = self.att_block(x, mask=causal_mask)
        x = self.layernorm1(x + pre_x)
        pre_x = tf.identity(x)
        x = self.ffn_block(x)
        out = self.layernorm2(x + pre_x)
        return out
    


class GPT(keras.Model):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout, layer_num, vocab_size, eps=1e-5, **kwargs):
        super(GPT, self).__init__(**kwargs)
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.eps = eps
        self.embedding = keras.layers.Embedding(vocab_size, d_model)
        self.out = keras.layers.Dense(vocab_size, activation='softmax')
        self.dropout = keras.layers.Dropout(dropout)
        self.transformer_layers = [AutoRgressiveLayer(d_model, ffn_hidden, num_heads, dropout, eps)
                       for _ in range(layer_num)]

        
    def position_encoding(self, d_model, max_sequence_length):
        even_i = tf.range(start=0, limit=d_model, delta=2, dtype=tf.float32)
        denominator = tf.pow(10000.0, even_i/d_model)
        position = tf.reshape( tf.range(max_sequence_length), shape=(max_sequence_length, 1))
        denominator = tf.cast(denominator, dtype=tf.float32)
        position = tf.cast(position, dtype=tf.float32)
        even_PE = tf.sin(position / denominator)
        odd_PE = tf.cos(position / denominator)
        stacked = tf.stack([even_PE, odd_PE], axis=2)
        return tf.cast(
            tf.reshape(stacked, shape=(max_sequence_length, d_model)), dtype=tf.float32
        )
    
    def create_masks(self, x_batch, NEG_INFTY=-1e9):
        num_sentences = len(x_batch)
        max_sequence_length = x_batch.shape[1]
        look_ahead_mask = np.full([max_sequence_length, max_sequence_length] , True)
        look_ahead_mask = np.triu(look_ahead_mask, k=1)
        decoder_padding_mask_self_attention = np.full([num_sentences, max_sequence_length, max_sequence_length] , False)

        for idx in range(num_sentences):
            x_batch_sentence_length = len(x_batch[idx])
            x_chars_to_padding_mask = np.arange(x_batch_sentence_length + 1, max_sequence_length)
            decoder_padding_mask_self_attention[idx, :, x_chars_to_padding_mask] = True
            decoder_padding_mask_self_attention[idx, x_chars_to_padding_mask, :] = True
    
        decoder_self_attention_mask =  np.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
        return tf.constant(value=decoder_self_attention_mask)

        
    def call(self, x: tf.Tensor):
        causal_mask = self.create_masks(x)
        x = self.embedding(x)
        batch_size, max_sequence_length, input_dims = x.shape
        pos = self.position_encoding(self.d_model, max_sequence_length)
        x = self.dropout(x + pos)
        for layer in self.transformer_layers:
            x = layer(x, causal_mask=causal_mask)
        return self.out(x)

if __name__ == "__main__":
    test_model = GPT(
        d_model=512,
        ffn_hidden=1024,
        num_heads=8,
        dropout=0.1,
        layer_num=6,
        vocab_size=6
    )
    x_test = tf.constant([[1, 2, 3, 4, 5, 0, 0, 0]])
    outputs_test = test_model(x_test)
    [
        print("đầu ra kiểm thử"),
        [print("-", end="") for _ in range(len("đầu ra kiểm thử"))],
        print()
    ]
    print(outputs_test)