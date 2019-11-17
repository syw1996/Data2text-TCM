""" Generator module """
import torch
import torch.nn as nn

import onmt.inputters as inputters
from onmt.utils.misc import aeq
from onmt.utils.loss import LossComputeBase


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks (See et al., 2017)
    (https://arxiv.org/abs/1704.04368), which consider copying words
    directly from the source sequence.

    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary

    """

    def __init__(self, input_size, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, len(tgt_dict))
        self.linear_copy = nn.Linear(input_size, 1)
        self.tgt_dict = tgt_dict

    def forward(self, hidden, attn, src_map, use_copy=True):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.tgt_dict.stoi[inputters.PAD_WORD]] = -float('inf')
        prob = torch.softmax(logits, 1)
        if not use_copy:
            return prob

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy.expand_as(prob))
        mul_attn = torch.mul(attn, p_copy.expand_as(attn))
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        # [batch*tlen, slen]
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorLoss(nn.Module):
    """ Copy generator criterion """

    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        scores (FloatTensor): (batch_size*tgt_len) x dynamic vocab size
        align (LongTensor): (batch_size*tgt_len)
        target (LongTensor): (batch_size*tgt_len)
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze()

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze()
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(LossComputeBase):
    """
    Copy Generator Loss Computation.
    """

    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length):
        super(CopyGeneratorLossCompute, self).__init__(criterion, generator)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

    def _make_shard_state(self, batch, output, back_output, ref_output, range_, ref_range_, attns, back_attns, ref_attns):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")
        
        return {
            "output": output,
            "back_output": back_output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "ref_output": ref_output,
            "ref_target": batch.ref_tgt[ref_range_[0] + 1: ref_range_[1]],
            "copy_attn": attns.get("copy"),
            "back_copy_attn": back_attns.get("copy"),
            "ref_copy_attn": ref_attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]],
            "ref_align": batch.ref_alignment[ref_range_[0] + 1: ref_range_[1]]
        }

    def _compute_loss(self, lambda_, batch, output, back_output, target, ref_output, ref_target, copy_attn, back_copy_attn, ref_copy_attn, align, ref_align):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        ref_target = ref_target.view(-1)
        align = align.view(-1)
        ref_align = ref_align.view(-1)
        scores = self.generator(self._bottle(output),
                                self._bottle(copy_attn),
                                batch.src_map)
        if back_output is not None:
            back_scores = self.generator(self._bottle(back_output),
                                        self._bottle(back_copy_attn),
                                        batch.ref_src_map)
        ref_scores = self.generator(self._bottle(ref_output),
                                    self._bottle(ref_copy_attn),
                                    batch.ref_src_map)
        
        content_loss = self.criterion(scores, align, target)
        if back_output is not None:
            back_loss = self.criterion(back_scores, ref_align, ref_target)
        style_loss = self.criterion(ref_scores, ref_align, ref_target)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = inputters.TextDataset.collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # back translation
        if back_output is not None:
            back_scores_data = inputters.TextDataset.collapse_copy_ref_scores(
                self._unbottle(back_scores.clone(), batch.batch_size),
                batch, self.tgt_vocab, batch.dataset.ref_src_vocabs)
            back_scores_data = self._bottle(back_scores_data)
        
        # reference

        ref_scores_data = inputters.TextDataset.collapse_copy_ref_scores(
            self._unbottle(ref_scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.ref_src_vocabs)
        ref_scores_data = self._bottle(ref_scores_data)

        ref_target_data = ref_target.clone()
        ref_correct_mask = (ref_target_data == unk) & (ref_align != unk)
        ref_offset_align = ref_align[ref_correct_mask] + len(self.tgt_vocab)
        ref_target_data[ref_correct_mask] += ref_offset_align
        
        # Compute sum of perplexities for stats
        content_stats = self._stats(content_loss.sum().clone(), scores_data, target_data)
        style_stats = self._stats(style_loss.sum().clone(), ref_scores_data, ref_target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt.ne(self.padding_idx).sum(0).float()
            ref_tgt_lens = batch.ref_tgt.ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            content_loss = content_loss.view(-1, batch.batch_size).sum(0)
            if back_output is not None:
                back_loss = back_loss.view(-1, batch.batch_size).sum(0)
            style_loss = style_loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            content_loss = torch.div(content_loss, tgt_lens).sum()
            if back_output is not None:
                back_loss = torch.div(back_loss, ref_tgt_lens).sum()
            style_loss = torch.div(style_loss, ref_tgt_lens).sum()
        else:
            content_loss = content_loss.sum()
            if back_output is not None:
                back_loss = back_loss.sum()
            style_loss = style_loss.sum()
        
        if back_output is not None:
            loss = lambda_ * content_loss + lambda_ * back_loss + (1 - 2 * lambda_) * style_loss
        else:
            loss = lambda_ * content_loss + (1 - lambda_) * style_loss

        return loss, content_stats, style_stats
