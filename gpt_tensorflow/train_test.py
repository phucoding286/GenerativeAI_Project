import numpy as np
import tensorflow as tf
import keras
from gpt import GPT

# dữ liệu mẫu
x = tf.constant(
    [
        [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
        [5, 6, 7, 8, 9, 0, 0, 0, 0, 0]
    ]
)

# khởi tạo mô hình
test_model = GPT(
    d_model=256,
    ffn_hidden=512,
    num_heads=8,
    dropout=0.1,
    layer_num=3,
    vocab_size=11
)

# tham số training
epochs=100
batch_size=1

# các hàm tối ưu hóa
optimizer = keras.optimizers.AdamW(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# vòng lặp train
for epoch in range(epochs):
    step = 0
    total_loss = []

    with tf.GradientTape(persistent=True) as tape:
        for batch_start in range(0, x.shape[0], batch_size):
            batch_end = batch_start + batch_size
            x_train = x[batch_start:batch_end, :]

            predictions = test_model.call(x_train)
            loss = loss_fn(x_train, predictions)
            total_loss.append(loss)
            gradients = tape.gradient(loss, test_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, test_model.trainable_variables))
            
            print(f"step {step} with loss: {loss}")
    step += 1
    print(f"finally epoch {epoch} with mean loss: {np.mean(total_loss)}")