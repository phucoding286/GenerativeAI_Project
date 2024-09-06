import keras
import tensorflow as tf
import math


class TransformerDecoder(keras.layers.Layer):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout, eps=1e-5, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dims = d_model // num_heads
        self.eps = eps


    def compute_attention(self, x: tf.Tensor, mask: tf.Tensor):
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
    

    def build(self, input_shape):
        self.qkv_w = keras.layers.Dense(3 * self.d_model)
        self.qkv_out = keras.layers.Dense(self.d_model)
        self.dropout_attention_product_out = keras.layers.Dropout(self.dropout)
        self.ffn_w1 = keras.layers.Dense(self.ffn_hidden)
        self.ffn_hidden_dropout = keras.layers.Dropout(self.dropout)
        self.ffn_w2 = keras.layers.Dense(self.d_model, activation="relu")
        self.ffn_out_dropout = keras.layers.Dropout(self.dropout)
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=self.eps)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=self.eps)
        return super(TransformerDecoder, self).build(input_shape)


    def call(self, x: tf.Tensor, mask: tf.Tensor):
        pre_x = tf.identity(x)
        x = self.compute_attention(x, mask=mask)
        x = self.layernorm1(x + pre_x)
        pre_x = tf.identity(x)
        x = self.ffn_w1(x)
        x = self.ffn_hidden_dropout(x)
        x = self.ffn_w2(x)
        x = self.ffn_out_dropout(x)
        out = self.layernorm2(x + pre_x)
        return out
    


class GPT(keras.layers.Layer):
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
        self.transformer_layers = [TransformerDecoder(d_model, ffn_hidden, num_heads, dropout, eps)
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

        
    def call(self, x: tf.Tensor, mask=None):
        x = self.embedding(x)
        batch_size, max_sequence_length, input_dims = x.shape
        pos = self.position_encoding(self.d_model, max_sequence_length)
        x = self.dropout(x + pos)
        for layer in self.transformer_layers:
            x = layer(x=x, mask=mask)
        return self.out(x)
