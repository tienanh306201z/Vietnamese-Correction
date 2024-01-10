import torch
from torch.nn.functional import log_softmax, softmax
import numpy as np




def translate(src, model, max_seq_length=128, sos_token=1, eos_token=2):
    "data: Bxsrc_len"
    model.eval()
    device = src.device

    with torch.no_grad():
        # src = model.cnn(img)
        src = src.transpose(1, 0)  # src_len x B
        memory = model.forward_encoder(src)

        translated_sentence = [[sos_token] * src.shape[1]]
        char_probs = [[1] * src.shape[1]]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)

            #            output = model(img, tgt_inp, tgt_key_padding_mask=None)
            #            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

    return translated_sentence, char_probs
