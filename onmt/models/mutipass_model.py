""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn

class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
    
    def inference(self, memory_bank, memory_lengths, dec_out):
        mb_device = memory_bank.device
        # src_len, batch_size, _ = memory_bank.size()
        tgt_len, batch_size, _ = dec_out.size()
        start_token = 2
        end_token = 3
        beam_size = 1
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            start_token,
            dtype=torch.long,
            device=mb_device)
        dec_outs = list()
        length_record = memory_lengths.data.new(*memory_lengths.size()).zero_()
        dec_memory_lengths = memory_lengths.data.new(*memory_lengths.size()).zero_()
        for step in range(tgt_len):
            decoder_input = alive_seq[:, -1].view(1, -1, 1)
            dec_out, dec_attn = self.decoder(decoder_input, memory_bank,
                                      memory_lengths=memory_lengths, dec_memory_bank=None)
            dec_outs.append(dec_out.squeeze(dim=0))
            # attn = dec_attn["std"]
            # scores:[batch_size, vocab_size]
            scores = self.generator(dec_out.view(-1, dec_out.size(2)))
            batch_ids = scores.argmax(1)
            alive_seq = batch_ids.unsqueeze(dim=1)
            length_record += length_record.data.eq(0).type(torch.cuda.LongTensor) * batch_ids.data.eq(end_token).type(torch.cuda.LongTensor)
            dec_memory_lengths += step * dec_memory_lengths.data.eq(0).type(torch.cuda.LongTensor) * batch_ids.data.eq(end_token).type(torch.cuda.LongTensor)
            if int(length_record.sum()) == batch_size:
                break
        dec_memory_lengths += (1 - length_record) * len(dec_outs)
        for i in range(batch_size):
            # 开始就结束情况
            if int(dec_memory_lengths[i]) == 0:
                dec_memory_lengths[i] = 1
        dec_outs = torch.stack(dec_outs, dim=0)
        return dec_outs, dec_memory_lengths

    def forward(self, src, tgt, lengths):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        # memory_bank:[src_len x batch x hidden]
        self.decoder.init_state(src, memory_bank, enc_state)
        # pass one:
        # attn['std']:[tgt_len x batch x src_len]
        dec_out0, attns0 = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths, dec_memory_bank=None, dec_memory_lengths=None)
        # inf_out:[tgt_len x batch x hidden]
        self.decoder.init_state(src, memory_bank, enc_state)
        inf_out, dec_memory_lengths = self.inference(memory_bank, lengths, dec_out0)
        
        # pass two:
        self.decoder.init_state(src, memory_bank, enc_state)
        dec_out1, attns1 = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths, dec_memory_bank=inf_out.transpose(0, 1), dec_memory_lengths = dec_memory_lengths)
        return dec_out0, attns0, dec_out1, attns1