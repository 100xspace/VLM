12. with torch.no_grad():
13.     outputs = model(**inputs)
14. last_hidden_state = outputs.last_hidden_state
15. print(last_hidden_state.shape)  # [1, num_patches + 1, hidden_dim]
