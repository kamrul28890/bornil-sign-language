import torch

from bornil.train.checkpoint import load_training_state, save_training_state


def test_save_load_training_state_restores_model_optimizer_scheduler(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step)

    for _ in range(3):
        optimizer.zero_grad()
        loss = model(torch.randn(4, 3)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

    expected_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    expected_opt = optimizer.state_dict()
    expected_sched = scheduler.state_dict()

    path = tmp_path / "last.pt"
    save_training_state(
        path, epoch=7, model=model, optimizer=optimizer, scheduler=scheduler,
        best_bleu=0.42, elapsed_min=12.5,
    )

    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.0)
    optimizer.param_groups[0]["lr"] = 99.0
    scheduler.last_epoch = 999

    state = load_training_state(path, model, optimizer, scheduler, device="cpu")

    assert state == {"epoch": 7, "best_bleu": 0.42, "elapsed_min": 12.5}
    for key, expected in expected_model.items():
        assert torch.allclose(model.state_dict()[key], expected)
    assert optimizer.state_dict()["state"].keys() == expected_opt["state"].keys()
    assert optimizer.state_dict()["param_groups"] == expected_opt["param_groups"]
    assert scheduler.state_dict() == expected_sched
