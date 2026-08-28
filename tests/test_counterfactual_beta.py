from types import SimpleNamespace

import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import compute_self_distillation_loss


def test_beta_two_target_matches_two_residual_units():
    student = torch.log_softmax(torch.tensor([[[0.1, -0.2, 0.3]]]), dim=-1)
    teacher_real = torch.log_softmax(torch.tensor([[[0.7, -0.4, 0.2]]]), dim=-1)
    teacher_null = torch.log_softmax(torch.tensor([[[-0.1, 0.5, 0.0]]]), dim=-1)
    mask = torch.ones((1, 1))
    config = SimpleNamespace(
        full_logit_distillation=True,
        distillation_topk=None,
        distillation_add_tail=True,
        counterfactual_null_mode="mean_color",
        counterfactual_extrapolation_beta=2.0,
        alpha=0.0,
        is_clip=None,
    )

    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student[..., 0],
        teacher_log_probs=teacher_real[..., 0],
        response_mask=mask,
        self_distillation_config=config,
        student_all_log_probs=student,
        teacher_all_log_probs=teacher_real,
        teacher_null_all_log_probs=teacher_null,
    )

    expected_target = F.log_softmax(teacher_real + 2.0 * (teacher_real - teacher_null), dim=-1)
    expected_loss = F.kl_div(student, expected_target, reduction="none", log_target=True).sum(-1).mean()
    expected_tv = 0.5 * (expected_target.exp() - teacher_real.exp()).abs().sum(-1).mean()
    torch.testing.assert_close(loss, expected_loss)
    assert metrics["self_distillation/counterfactual_extrapolation_beta"] == 2.0
    assert abs(metrics["self_distillation/counterfactual_target_tv"] - expected_tv.item()) < 1e-7
