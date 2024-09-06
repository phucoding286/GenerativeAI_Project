import tensorflow as tf
import keras



loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = None



@tf.function
def train_step(x, y, model, mask):
    with tf.GradientTape() as tape:
        predictions = model.call(x, mask)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def training(x_batch, y_batch, model, mask, lr=0.0001, batch_size=10, epochs=1, step_verbose=False):
    global optimizer
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        step = 0
        for batch_start in range(0, x_batch.shape[0], batch_size):
            batch_end = batch_start + batch_size
            x_train = x_batch[batch_start:batch_end, :]
            y_train = y_batch[batch_start:batch_end, :]
            mask = mask[batch_start:batch_end, :, :]

            loss = train_step(x_train, y_train, model, mask)
            
            if step_verbose: print(f"step {step}, loss: {loss}")
            step += 1
        print(f"finally epoch {epoch}, loss: {loss}")
