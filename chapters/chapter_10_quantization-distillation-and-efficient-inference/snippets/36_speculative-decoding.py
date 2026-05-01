generated = input_ids.clone()
    while generated.shape[1] < max_length:
        # Step 1: Draft model proposes draft_steps tokens
        draft_input, draft_proposals = generated.clone(), []
        for _ in range(draft_steps):
            draft_logits = draft_model(draft_input).logits[:, -1, :]
            next_token = torch.argmax(draft_logits, dim=-1, keepdim=True)
            draft_proposals.append(next_token)
            draft_input = torch.cat([draft_input, next_token], dim=1)
        # Step 2: Target model verifies proposals (stop at first rejection)
        verification_input, accepted_tokens = generated.clone(), []
        for proposed_token in draft_proposals:
            target_logits = target_model(verification_input).logits[:, -1, :]
            target_probs = F.softmax(target_logits, dim=-1)
            target_top = torch.argmax(target_probs, dim=-1, keepdim=True)
            # Accept if target matches, or target assigns sufficient probability mass
            proposed_idx = proposed_token.item() if proposed_token.numel() == 1 else proposed_token
            if torch.equal(target_top, proposed_token) or target_probs[0, proposed_idx] > accept_p:
                accepted_tokens.append(proposed_token)
                verification_input = torch.cat([verification_input, proposed_token], dim=1)
            else:
                accepted_tokens.append(target_top)
