generated = torch.cat([generated] + accepted_tokens, dim=1)
    return generated
