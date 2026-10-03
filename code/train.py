"""train.py — Seed, đánh giá, huấn luyện, dự đoán và ghi tệp nộp.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:  # Works both from ``code/`` in the notebook and as package ``code.train``.
    from .data import iterate_batches
    from .model import MLP, EXPECTED_PARAMS, count_params
    from .optimizer import build_optimizer, clip_gradients
except ImportError:  # pragma: no cover - used by a notebook with code/ on sys.path
    from data import iterate_batches
    from model import MLP, EXPECTED_PARAMS, count_params
    from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Điểm xuất phát; chỉ tinh chỉnh bằng validation, không dùng eval.
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm)
    if cm.shape != (7, 7):
        raise ValueError(f"cm phải có shape (7, 7), nhận được {cm.shape}")
    true_positive = np.diag(cm).astype(np.float64)
    precision_denominator = cm.sum(axis=0)
    recall_denominator = cm.sum(axis=1)
    precision = np.divide(true_positive, precision_denominator,
                          out=np.zeros(7, dtype=np.float64), where=precision_denominator != 0)
    recall = np.divide(true_positive, recall_denominator,
                       out=np.zeros(7, dtype=np.float64), where=recall_denominator != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(7, dtype=np.float64), where=(precision + recall) != 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    predictions = []
    for start in range(0, len(X), batch_size):
        logits = model(X[start:start + batch_size])
        predictions.append(logits.argmax(dim=1))
    return torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.int64, device=X.device)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if len(X) != len(y):
        raise ValueError("X và y phải có cùng số mẫu")
    if len(X) == 0:
        raise ValueError("Không thể evaluate tập rỗng")
    model.eval()
    total_loss = 0.0
    all_predictions = []
    for start in range(0, len(X), batch_size):
        xb, yb = X[start:start + batch_size], y[start:start + batch_size]
        logits = model(xb)
        if loss_name == "ce":
            total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        elif loss_name == "mse":
            one_hot = F.one_hot(yb, num_classes=7).to(dtype=logits.dtype)
            total_loss += F.mse_loss(logits, one_hot, reduction="sum").item()
        else:
            raise ValueError("loss_name phải là 'ce' hoặc 'mse'")
        all_predictions.append(logits.argmax(dim=1))

    predictions = torch.cat(all_predictions)
    accuracy = (predictions == y).float().mean().item()
    encoded = y.to(torch.int64) * 7 + predictions.to(torch.int64)
    confusion = torch.bincount(encoded, minlength=49).reshape(7, 7).cpu().numpy()
    # MSELoss averages over both samples and seven output coordinates; CE averages samples only.
    divisor = len(X) * (7 if loss_name == "mse" else 1)
    return {"loss": total_loss / divisor, "acc": accuracy, "macro_f1": macro_f1_from_confusion(confusion)}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        one_hot = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, one_hot)
    raise ValueError("loss_name phải là 'ce' hoặc 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    effective_cfg = {**DEFAULT_CFG, **cfg}
    if effective_cfg["lr"] is None:
        raise ValueError("Hãy đặt cfg['lr'] bằng kết quả chọn trên validation")
    if effective_cfg["epochs"] <= 0 or effective_cfg["batch"] <= 0:
        raise ValueError("epochs và batch phải lớn hơn 0")
    if effective_cfg["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision phải là 'fp32', 'fp16', hoặc 'bf16'")

    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    if X_val.device != device or y_tr.device != device or y_val.device != device:
        raise ValueError("Các tensor train/val phải nằm trên cùng một device")
    if effective_cfg["precision"] != "fp32" and device.type != "cuda":
        raise ValueError("fp16/bf16 chỉ được hỗ trợ trong pipeline này trên CUDA")
    if effective_cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU hiện tại không hỗ trợ BF16")

    set_seed(effective_cfg["seed"])
    hidden = tuple(effective_cfg["hidden"])
    model = MLP(hidden=hidden, dropout=effective_cfg["dropout"], init=effective_cfg["init"]).to(device)
    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden], "Số tham số model không đúng kiến trúc quy định"
    optimizer = build_optimizer(
        effective_cfg["optimizer"], model.parameters(), effective_cfg["lr"],
        weight_decay=effective_cfg["weight_decay"], momentum=effective_cfg["momentum"],
    )
    precision = effective_cfg["precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16")) if device.type == "cuda" else None
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    generator = torch.Generator(device=device).manual_seed(effective_cfg["seed"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    step0_loss = evaluate(model, X_val, y_val, effective_cfg["loss"])["loss"]
    history = {key: [] for key in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm", "epoch_time_s")}
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss, best_epoch = float("inf"), 0
    diverged = False

    for epoch in range(1, effective_cfg["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        model.train()
        gradient_norms = []
        for xb, yb in iterate_batches(X_tr, y_tr, effective_cfg["batch"], generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model(xb)
                loss = compute_loss(logits, yb, effective_cfg["loss"])
            if not torch.isfinite(loss):
                diverged = True
                break
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norms.append(clip_gradients(model.parameters(), effective_cfg["clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gradient_norms.append(clip_gradients(model.parameters(), effective_cfg["clip_norm"]))
                optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started_at
        if diverged:
            break

        train_metrics = evaluate(model, X_tr, y_tr, effective_cfg["loss"])
        val_metrics = evaluate(model, X_val, y_val, effective_cfg["loss"])
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(np.mean(gradient_norms)) if gradient_norms else float("nan"))
        history["epoch_time_s"].append(elapsed)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss, best_epoch = val_metrics["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    best_index = best_epoch - 1
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss if best_epoch else float("nan"),
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "val_acc": history["val_acc"][best_index] if best_epoch else float("nan"),
        "val_macro_f1": history["val_macro_f1"][best_index] if best_epoch else float("nan"),
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else float("nan"),
        "peak_mem_MB": peak_memory,
        "diverged": diverged,
    }
    return {"cfg": effective_cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id)
    preds = np.asarray(preds)
    if row_id.ndim != 1 or preds.ndim != 1 or len(row_id) != len(preds):
        raise ValueError("row_id và preds phải là hai mảng 1 chiều có cùng số phần tử")
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("row_id phải duy nhất")
    if not np.all(np.equal(preds, np.floor(preds))):
        raise ValueError("preds phải chứa giá trị nguyên")
    if not np.all((preds >= 0) & (preds <= 6)):
        raise ValueError("preds phải là nhãn nguyên trong khoảng 0..6")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(row_id.astype(np.int64), preds.astype(np.int64)))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    final_cfg = result.get("cfg", cfg)
    device = data["X_eval"].device
    model = MLP(hidden=tuple(final_cfg["hidden"]), dropout=final_cfg["dropout"], init=final_cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    predictions = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], predictions.cpu().numpy(), pred_path)
