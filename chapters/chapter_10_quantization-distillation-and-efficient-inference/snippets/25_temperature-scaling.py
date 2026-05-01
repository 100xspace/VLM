for size in intermediate_sizes + [final_size]:
        print(f"\nDistilling to {size}B parameter model...")
        # Initialize student with size `size`
        student = initialize_vlm(num_params=size * 1e9)
        # Distill current teacher -> student
        student = train_distilled_vlm(teacher=current_teacher, student=student, train_loader=train_loader, num_epochs=5)
        # Evaluate
        accuracy = evaluate_model(student)
        print(f"{size}B model accuracy: {accuracy:.2f}%")
        # Student becomes next teacher
        current_teacher = student
    return current_teacher
# Example: 70B -> 7B -> 3B distillation chain
# final_model = progressive_distillation(
#     teacher_model=vlm_70b,
#     intermediate_sizes=[13, 7],
#     final_size=3,
#     train_loader=train_data
# )
