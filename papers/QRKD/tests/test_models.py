from __future__ import annotations

import torch
from lib.models import StudentCNN, TeacherCNN
from lib.utils import count_parameters


def test_student_cnn_output_shapes_and_params():
    model = StudentCNN()
    x = torch.randn(2, 1, 28, 28)
    logits, features = model(x)
    assert logits.shape == (2, 10)
    assert features.shape[0] == 2
    assert count_parameters(model) == 1725


def test_teacher_cnn_has_expected_param_count():
    teacher = TeacherCNN()
    assert count_parameters(teacher) == 6690
