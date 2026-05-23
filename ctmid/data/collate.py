from __future__ import annotations

import torch


def sequential_collate_fn(samples: list[dict]) -> dict[str, torch.Tensor | list[list[int]]]:
    batch_size = len(samples)
    max_len = max((sample["seq_len"] for sample in samples), default=1)
    item_seq = torch.zeros(batch_size, max_len, dtype=torch.long)
    time_seq = torch.zeros(batch_size, max_len, dtype=torch.float)
    seq_len = torch.zeros(batch_size, dtype=torch.long)
    target_item = torch.zeros(batch_size, dtype=torch.long)
    user_id = torch.zeros(batch_size, dtype=torch.long)
    history_items: list[list[int]] = []
    for row, sample in enumerate(samples):
        length = int(sample["seq_len"])
        seq_len[row] = length
        target_item[row] = int(sample["target_item"])
        user_id[row] = int(sample.get("user_id", 0))
        history_items.append(list(sample.get("history_items", [])))
        if length:
            item_seq[row, :length] = torch.tensor(sample["item_seq"], dtype=torch.long)
            time_seq[row, :length] = torch.tensor(sample["time_seq"], dtype=torch.float)
    return {
        "user_id": user_id,
        "item_seq": item_seq,
        "time_seq": time_seq,
        "seq_len": seq_len,
        "target_item": target_item,
        "history_items": history_items,
    }
