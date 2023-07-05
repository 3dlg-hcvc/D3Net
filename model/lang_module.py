
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LangModule(nn.Module):
    def __init__(self, cfg, 
        emb_size=300, hidden_size=256):
        super().__init__() 

        self.num_text_classes = cfg.model.num_bbox_class
        self.use_lang_classifier = cfg.model.use_lang_classifier
        self.use_bidir = cfg.model.use_bidir

        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )

        # language classifier
        if self.use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(self.hidden_size, self.num_text_classes),
                nn.Dropout()
            )


    def forward(self, data_dict):
        """
        encode the input descriptions
        """

        word_embs = data_dict["lang_feat"] # batch_size, chunk_size, seq_len, emb_size
        batch_size, chunk_size, seq_len, _ = word_embs.shape
        word_embs = word_embs.reshape(-1, seq_len, self.emb_size) # batch_size * chunk_size, seq_len, emb_size

        lang_len = data_dict["lang_len"] # batch_size, chunk_size
        lang_len = lang_len.reshape(-1) # batch_size * chunk_size

        lang_feat = pack_padded_sequence(word_embs, lang_len.cpu(), batch_first=True, enforce_sorted=False)

        # encode description
        lang_hiddens, lang_last = self.gru(lang_feat)
        lang_hiddens, _ = pad_packed_sequence(lang_hiddens, batch_first=True)
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size * chunk_size, hidden_size * num_dir

        if self.use_bidir:
            lang_hiddens = (lang_hiddens[:, :, :int(lang_hiddens.shape[-1] / 2)] + lang_hiddens[:, :, int(lang_hiddens.shape[-1] / 2):]) / 2
            lang_last = (lang_last[:, :int(lang_last.shape[-1] / 2)] + lang_last[:, int(lang_last.shape[-1] / 2):]) / 2

        assert lang_hiddens.shape[-1] == self.hidden_size
        assert lang_last.shape[-1] == self.hidden_size

        # HACK zero-padding hiddens
        pad_hiddens = torch.zeros(batch_size * chunk_size, seq_len, self.hidden_size).type_as(lang_hiddens)
        pad_hiddens[:, :lang_hiddens.shape[1]] = lang_hiddens

        # sentence mask
        lengths = lang_len.unsqueeze(1).repeat(1, seq_len) # batch_size * chunk_size, seq_len
        idx = torch.arange(0, seq_len).unsqueeze(0).repeat(lengths.shape[0], 1).type_as(lengths).long() # batch_size * chunk_size, seq_len
        lang_masks = (idx < lengths).float() # batch_size * chunk_size, seq_len
        data_dict["lang_masks"] = lang_masks # batch_size * chunk_size, seq_len

        # store the encoded language features
        data_dict["lang_hiddens"] = pad_hiddens # batch_size * chunk_size, seq_len, hidden_size
        data_dict["lang_emb"] = lang_last # batch_size * chunk_size, hidden_size

        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"]) # batch_size * chunk_size, num_text_classes

        return data_dict

