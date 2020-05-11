import torch
import torch.nn as nn


class RelationClassifierModel(nn.Module):
    def __init__(self, spec):
        super(RelationClassifierModel, self).__init__()

        self.spec = spec

        enc_size = self._obj_enc_size() * 2
        feat1 = 1024
        feat2 = 256
        feat3 = 64
        dropout = 0.2

        self.encoder = nn.Sequential(
            nn.Linear(enc_size, feat1),
            nn.Dropout(dropout),
            nn.Linear(feat1, feat2),
            nn.Dropout(dropout),
            nn.Linear(feat2, feat3)
        )

        self.rel_layers = {}
        for rel in spec.relations:
            rel_layer = nn.Linear(feat3, 1)
            self.add_module(rel, rel_layer)
            self.rel_layers[rel] = rel_layer

    def forward(self, x):
        feat = self.encoder(x)

        rel_outs = {}
        for rel in self.spec.relations:
            rel_out = self.rel_layers[rel](feat)
            rel_out = torch.sigmoid(rel_out)
            rel_outs[rel] = rel_out

        return rel_outs

    def _obj_enc_size(self):
        enc_size = 4
        enc_size += len(self.spec.obj_types())
        for prop in self.spec.prop_names():
            enc_size += len(self.spec.prop_values(prop))

        return enc_size
