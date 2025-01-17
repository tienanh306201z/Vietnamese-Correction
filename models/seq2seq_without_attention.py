import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: src_len x batch_size
        outputs: src_len x batch_size x hid_dim
        hidden: batch_size x hid_dim
        """

        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        """
        inputs: batch_size
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """

        input = input.unsqueeze(0)  # 1 x B

        embedded = self.dropout(self.embedding(input))

        # a = self.attention(hidden, encoder_outputs)

        # a = a.unsqueeze(1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # weighted = torch.bmm(a, encoder_outputs)

        # weighted = weighted.permute(1, 0, 2)

        # rnn_input = torch.cat((embedded, weighted), dim = 2)
        rnn_input = embedded

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # assert (output == hidden).all()

        # embedded = embedded.squeeze(0)
        # output = output.squeeze(0)
        # weighted = weighted.squeeze(0)

        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden.squeeze(0), None


class Seq2Seq_WithoutAtt(nn.Module):
    def __init__(self, input_dim, output_dim, encoder_embbeded, decoder_embedded, encoder_hidden, decoder_hidden,
                 encoder_dropout=0.1, decoder_dropout=0.1):
        super().__init__()

        # attn = Attention(encoder_hidden, decoder_hidden)

        self.encoder = Encoder(input_dim, encoder_embbeded, encoder_hidden, decoder_hidden, encoder_dropout)
        self.decoder = Decoder(output_dim, decoder_embedded, encoder_hidden, decoder_hidden, decoder_dropout)

    def forward_encoder(self, src):
        """
        src: timestep x batch_size x channel
        hidden: batch_size x hid_dim
        encoder_outputs: src_len x batch_size x hid_dim
        """

        encoder_outputs, hidden = self.encoder(src)

        return (hidden, encoder_outputs)

    def forward_decoder(self, tgt, memory):
        """
        tgt: timestep x batch_size
        hidden: batch_size x hid_dim
        encouder: src_len x batch_size x hid_dim
        output: batch_size x 1 x vocab_size
        """
        tgt = tgt[-1]
        hidden, encoder_outputs = memory
        output, hidden, _ = self.decoder(tgt, hidden, encoder_outputs)
        output = output.unsqueeze(1)

        return output, (hidden, encoder_outputs)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: time_step x batch_size
        trg: time_step x batch_size
        outputs: batch_size x time_step x vocab_size
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        device = src.device

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)
        encoder_outputs, hidden = self.encoder(src)

        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)

            # Teacher force
            rnd_teacher = torch.rand(1).item()
            teacher_force = rnd_teacher < teacher_forcing_ratio
            input = trg[t, :] if teacher_force else top1
            # print("teach ratio: ", teacher_forcing_ratio)
            # print("input teaching: ", input, " ratio: ", teacher_force, rnd_teacher, teacher_forcing_ratio, sep=" || ")
            # print("trg[t,:]", trg[t, :], " \n top1 :", top1)

        outputs = outputs.transpose(0, 1).contiguous()
        return outputs

    # def expand_memory(self, memory, beam_size):
    #     hidden, encoder_outputs = memory
    #     hidden = hidden.repeat(beam_size, 1)
    #     encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)

    #     return (hidden, encoder_outputs)

    def get_memory(self, memory, i):
        hidden, encoder_outputs = memory
        hidden = hidden[[i]]
        encoder_outputs = encoder_outputs[:, [i], :]

        return (hidden, encoder_outputs)