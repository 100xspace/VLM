for r in responses:
        if ocr_tokens is not None:
            for word in r.split():
                if word.isupper() and word not in ocr_tokens:
                hallucinated.append(r)
        if any(brand in r.lower() for brand in ["apple", "dell", "hp"]):
        hallucinated.append(r)
    return hallucinated
