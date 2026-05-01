def progressive_distillation(teacher_model, intermediate_sizes=[70, 34, 13, 7], final_size=3, train_loader=None):
    """
    Progressive distillation: teacher -> large student -> medium student -> small student
    Args:
        teacher_model: Original large model (e.g., 70B parameters)
        intermediate_sizes: List of intermediate model sizes in billions
        final_size: Final target size in billions
        train_loader: Training data
    Returns:
