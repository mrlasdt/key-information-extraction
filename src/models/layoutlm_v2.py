from transformers import LayoutLMv2ForTokenClassification
from torch import nn


class LayoutLMv2ForKIE(LayoutLMv2ForTokenClassification):
    def __init__(self, pretrained, **kwargs):
        super().from_pretrained(pretrained, **kwargs)

    @classmethod
    def from_pretrained_with_new_head(pretrained, new_num_labels, **kwargs):
        model = LayoutLMv2ForKIE(pretrained, **kwargs)
        model.classifier = nn.Linear(model.config.hidden_size, new_num_labels)
        model.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        if model.classifier.bias is not None:
            model.classifier.bias.data.zero_()
        return model
