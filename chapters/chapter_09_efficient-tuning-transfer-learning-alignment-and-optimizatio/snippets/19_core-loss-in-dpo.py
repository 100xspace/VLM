def dpo_loss(
policy_chosen_logp,
policy_rejected_logp,
ref_chosen_logp,
ref_rejected_logp,
beta=0.1
):
    """
