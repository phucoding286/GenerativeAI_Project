import numpy as np

NEG_INFTY = -1e9

def create_masks(x_batch, look_ahead_mask_c=True):
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
    
    if look_ahead_mask_c:
        decoder_self_attention_mask =  np.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    else:
       decoder_self_attention_mask =  np.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    return decoder_self_attention_mask