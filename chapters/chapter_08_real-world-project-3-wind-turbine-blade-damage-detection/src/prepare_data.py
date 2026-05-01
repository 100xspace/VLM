def get_crop_center_with_randomness(
    position, x_min, y_min, x_max, y_max, crop_size, min_margin=0.05, max_margin=0.2
):
    margin_frac_x = random.uniform(min_margin, max_margin)
    margin_frac_y = random.uniform(min_margin, max_margin)
    margin_x = margin_frac_x * crop_size
    margin_y = margin_frac_y * crop_size
    if position == "center":
        crop_cx = (x_min + x_max) / 2
        crop_cy = (y_min + y_max) / 2
    elif position == "top-left":
        crop_cx = x_min + margin_x
        crop_cy = y_min + margin_y
    return crop_cx, crop_cy
