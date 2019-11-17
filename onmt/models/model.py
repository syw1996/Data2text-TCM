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

    def __init__(self, src_encoder, tgt_encoder, decoder, co_attention=None, generator=None):
        super(NMTModel, self).__init__()
        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder
        self.decoder = decoder
        self.co_attention = co_attention
        self.generator = generator
    
    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))
    
    def inference(self, dec_outs, attns, src_map):
        """
        inference output words from z (x + y' = z)
        Args:
            dec_out (`Tensor`): decoder output
                `[tgt_len x batch x hidden]`.
        
        Returns:
            inf_seq (`Tensor`): 
                `[tgt_len x batch_size]`.
        """
        tgt_len = dec_outs.size(0)
        inf_lst = []
        for step in range(tgt_len):
            dec_out = dec_outs[step]
            attn = attns[step]
            # scores:[batch_size, vocab_size]
            scores = self.generator(dec_out, attn, src_map, False)
            batch_ids = scores.argmax(1)
            inf_lst.append(batch_ids)
        inf = torch.stack(inf_lst)
        
        return inf.unsqueeze(2)


    def forward(self, src, src_map, ref_src, tgt, ref_tgt, mask_ref_tgt, lengths, ref_src_lengths, ref_tgt_lengths):
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
        ref_tgt = ref_tgt[:-1] # exclude last target from inputs

        enc_state, memory_bank, lengths = self.src_encoder(src, lengths)
        ref_enc_state, ref_memory_bank, ref_src_lengths = self.src_encoder(ref_src, ref_src_lengths)

        _, tgt_memory_bank, _ = self.tgt_encoder(mask_ref_tgt, ref_tgt_lengths)

        if self.co_attention is not None:
            org_tgt_memory_bank = self.co_attention(enc_state, memory_bank, tgt_memory_bank, lengths, ref_tgt_lengths)
            ref_tgt_memory_bank = self.co_attention(ref_enc_state, ref_memory_bank, tgt_memory_bank, ref_src_lengths, ref_tgt_lengths)
        else:
            org_tgt_memory_bank = tgt_memory_bank
            ref_tgt_memory_bank = tgt_memory_bank
        
        # content generate: x + y' = y
        self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank, org_tgt_memory_bank,
                                      memory_lengths=lengths)

        # back translation: x + y' = z; x' + z = y' 
        if self.generator is not None:      
            inf = self.inference(dec_out, attns.get("copy"), src_map)
            inf = inf[:-1]
            _, inf_memory_bank, _ = self.tgt_encoder(inf, None)
            if self.co_attention is not None:
                inf_memory_bank_ = self.co_attention(ref_enc_state, ref_memory_bank, inf_memory_bank, ref_src_lengths, None)
            else:
                inf_memory_bank_ = inf_memory_bank
            self.decoder.init_state(ref_src, ref_memory_bank, ref_enc_state)
            back_dec_out, back_attns = self.decoder(ref_tgt, ref_memory_bank, inf_memory_bank_,
                                        memory_lengths=ref_src_lengths)
        else:
            back_dec_out = back_attns = None
        
        # style transform: x' + y' = y'
        self.decoder.init_state(ref_src, ref_memory_bank, ref_enc_state)
        ref_dec_out, ref_attns = self.decoder(ref_tgt, ref_memory_bank, ref_tgt_memory_bank,
                                      memory_lengths=ref_src_lengths)

        return dec_out, attns, back_dec_out, back_attns, ref_dec_out, ref_attns
