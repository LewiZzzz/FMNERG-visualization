import pytorch_lightning as pl

class T5Inference(pl.LightningModule):
    """
    A simplified version of T5FineTuner for inference only.
    """

    def __init__(self, tfm_model, tokenizer):
        super().__init__()
        self.model = tfm_model
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None, vis_feats=None, vis_attention_mask=None):
        """
        Forward pass for inference.
        """
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            vis_feats=vis_feats,
            vis_attention_mask=vis_attention_mask,
        )

    def generate(self, input_ids, attention_mask=None, vis_feats=None, vis_attention_mask=None, max_length=200, num_beams=1):
        """
        Generate predictions using the model.
        """
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vis_feats": vis_feats,
            "vis_attention_mask": vis_attention_mask,
            "max_length": max_length,
            "num_beams": num_beams,
        }

        print("(generate vis_feats):", model_kwargs["vis_feats"])

        return self.model.generate(**model_kwargs)

