
from transformers import LayoutLMv2ForTokenClassification
PRETRAINED_MODEL = "microsoft/layoutxlm-base"


def load_model(pretrained_model, kie_labels, device):
    return LayoutLMv2ForTokenClassification.from_pretrained(
        pretrained_model, num_labels=len(kie_labels)).to(
        device)
