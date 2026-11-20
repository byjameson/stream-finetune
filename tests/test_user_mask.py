import torch

from streamft.spiking_cli import mask_user_labels


def test_mask_user_labels_keeps_assistant_tokens():
    # 0=user, 1=assistant, 2=end
    y = torch.tensor([[0, 5, 6, 1, 7, 8, 2, 0, 9, 1, 10, 2]])
    masked = mask_user_labels(y, 0, 1)
    # user token + user text masked; assistant tag/text/end kept
    assert (masked[0, :3] == -100).all()
    assert masked[0, 3].item() == 1
    assert masked[0, 4].item() == 7
    assert masked[0, 5].item() == 8
    assert masked[0, 6].item() == 2
    # second user turn masked, assistant turn kept
    assert masked[0, 7].item() == -100
    assert masked[0, 8].item() == -100
    assert masked[0, 9].item() == 1
    assert masked[0, 10].item() == 10
    assert masked[0, 11].item() == 2
