"""Restricted fine-tuning of an existing WiSAR detector."""
def freeze_for_heatmap_finetuning(model):
    """Keep geometry and feature statistics fixed while learning class scores."""
    if not getattr(model, "class_regression", False) or model.num_classes != 22:
        raise ValueError("heatmap fine-tuning requires the 22-class WiSAR model")
    alpha = getattr(model, "reg_gate_alpha", None)
    if alpha is not None and float(alpha.detach()) > 0:
        raise ValueError("active heatmap regression gate would change geometry")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("hm_head."))
    # Both BatchNorm updates and feature dropout must remain disabled.
    model.eval()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
