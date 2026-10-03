"""plots.py — Vẽ biểu đồ riêng và biểu đồ so sánh cho thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """
    cfg = result["cfg"]
    history = result["history"]
    required = {"epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm"}
    missing = required.difference(history)
    if missing:
        raise ValueError(f"history thiếu các khoá: {sorted(missing)}")
    epochs = history["epoch"]
    if not epochs:
        raise ValueError("Không thể vẽ một lần chạy chưa hoàn thành epoch nào")

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    best_epoch = result.get("summary", {}).get("best_epoch")
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val loss")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], marker="o", label="val accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], marker="o", label="val macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()

    axes[2].plot(epochs, history["grad_norm"], marker="o", label="pre-clip grad norm")
    axes[2].set(title="Gradient norm (before clipping)", xlabel="Epoch", ylabel="L2 norm")
    axes[2].legend()
    if best_epoch:
        for axis in axes:
            axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.55, label="best epoch")
            axis.grid(alpha=0.25)

    figure.suptitle(
        f"{cfg['exp_id']} | {cfg['optimizer']}, lr={cfg['lr']}, batch={cfg['batch']}, "
        f"dropout={cfg['dropout']}, precision={cfg['precision']}"
    )
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("Cần ít nhất một result để vẽ so sánh")
    figure, axis = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result["history"]
        if metric not in history:
            raise ValueError(f"{result['cfg'].get('exp_id', '<unknown>')} không có metric {metric!r}")
        axis.plot(history["epoch"], history[metric], marker="o", label=result["cfg"]["exp_id"])
    axis.set(
        title=title or f"Comparison: {metric}", xlabel="Epoch",
        ylabel=metric.replace("_", " "),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
