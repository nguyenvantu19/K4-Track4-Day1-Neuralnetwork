"""data.py — Nạp, tách, chuẩn hoá và chia lô dữ liệu Forest CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations  # Cho phép dùng kiểu chú thích hiện đại, kể cả khi kiểu đó được khai báo ở dưới.

import numpy as np  # Nhập NumPy để thao tác với mảng dữ liệu.
import torch  # Nhập PyTorch để tạo tensor và huấn luyện trên CPU/GPU.
from sklearn.model_selection import train_test_split  # Dùng hàm có sẵn để tách train/validation mà vẫn giữ tỉ lệ lớp.

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):  # Định nghĩa hàm nạp hai tập dữ liệu đã chia sẵn.
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_path = f"{processed_dir}/train.npz"  # Tạo đường dẫn đến file chứa toàn bộ tập train.
    eval_path = f"{processed_dir}/eval.npz"  # Tạo đường dẫn đến file chứa tập eval dùng để chấm cuối.

    with np.load(train_path) as train_data:  # Mở file train và tự đóng file sau khi đã đọc xong.
        X_train_full = train_data["X"]  # Lấy ma trận đặc trưng của tập train.
        y_train_full = train_data["y"]  # Lấy mảng nhãn tương ứng của tập train.

    with np.load(eval_path) as eval_data:  # Mở file eval và tự đóng file sau khi đã đọc xong.
        X_eval = eval_data["X"]  # Lấy ma trận đặc trưng của tập eval.
        y_eval = eval_data["y"]  # Lấy nhãn thật của tập eval; không dùng nhãn này để chọn mô hình.
        eval_row_id = eval_data["row_id"]  # Lấy mã dòng để ghi dự đoán đúng thứ tự khi nộp bài.

    for X, y, split_name in ((X_train_full, y_train_full, "train"), (X_eval, y_eval, "eval")):  # Kiểm tra lần lượt hai tập theo cùng quy ước.
        assert X.ndim == 2 and X.shape[1] == 54, f"{split_name}: X phải có shape (N, 54), nhận được {X.shape}"  # X phải là ma trận 54 đặc trưng.
        assert X.dtype == np.float32, f"{split_name}: X phải có dtype float32, nhận được {X.dtype}"  # Đảm bảo đặc trưng đúng kiểu số thực dùng cho PyTorch.
        assert y.ndim == 1 and y.shape[0] == X.shape[0], f"{split_name}: y phải có shape ({X.shape[0]},), nhận được {y.shape}"  # Mỗi mẫu X phải có đúng một nhãn.
        assert y.dtype == np.int64, f"{split_name}: y phải có dtype int64, nhận được {y.dtype}"  # CrossEntropyLoss yêu cầu nhãn kiểu int64.
        assert np.all((0 <= y) & (y <= 6)), f"{split_name}: nhãn phải thuộc khoảng 0..6"  # Kiểm tra nhãn đã được mã hoá cho 7 lớp.

    assert eval_row_id.ndim == 1 and eval_row_id.shape[0] == X_eval.shape[0], f"eval: row_id phải có shape ({X_eval.shape[0]},), nhận được {eval_row_id.shape}"  # Mỗi mẫu eval cần một mã dòng tương ứng.
    assert eval_row_id.dtype == np.int64, f"eval: row_id phải có dtype int64, nhận được {eval_row_id.dtype}"  # Giữ mã dòng ở kiểu số nguyên nhất quán với file gốc.

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id  # Trả dữ liệu theo đúng thứ tự mà các hàm phía dưới sử dụng.


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):  # Định nghĩa hàm tách validation từ tập train.
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    assert 0 < val_fraction < 1, "val_fraction phải nằm trong khoảng (0, 1)"  # Đảm bảo tỉ lệ validation là hợp lệ.
    assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "X và y phải có cùng số mẫu"  # Kiểm tra đầu vào trước khi tách dữ liệu.
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_fraction, random_state=seed, stratify=y)  # Tách phân tầng để mỗi tập giữ gần đúng tỉ lệ các lớp ban đầu.
    return X_tr, y_tr, X_val, y_val  # Trả train trước, validation sau để dùng cho huấn luyện và đánh giá.


def fit_standardizer(X_tr):  # Định nghĩa hàm học thông số chuẩn hoá từ train.
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    assert X_tr.ndim == 2 and X_tr.shape[1] == 54, "X_tr phải có shape (N, 54)"  # Đảm bảo dữ liệu có đúng 54 đặc trưng.
    numeric_features = X_tr[:, :N_NUMERIC]  # Chọn riêng 10 cột số liên tục cần chuẩn hoá.
    mean = numeric_features.mean(axis=0, dtype=np.float32)  # Tính trung bình từng cột chỉ từ tập train.
    std = numeric_features.std(axis=0, dtype=np.float32)  # Tính độ lệch chuẩn từng cột chỉ từ tập train.
    std = np.where(std == 0, np.float32(1.0), std)  # Đổi độ lệch chuẩn bằng 0 thành 1 để không chia cho 0.
    return mean, std  # Trả các thống kê để áp dụng giống hệt cho train, validation và eval.


def apply_standardizer(X, mean, std):  # Định nghĩa hàm áp dụng chuẩn hoá mà không sửa dữ liệu gốc.
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    assert X.ndim == 2 and X.shape[1] == 54, "X phải có shape (N, 54)"  # Kiểm tra dữ liệu đầu vào có đúng số cột.
    assert mean.shape == (N_NUMERIC,) and std.shape == (N_NUMERIC,), "mean và std phải có 10 phần tử"  # Đảm bảo mỗi cột số có một mean và một std.
    X_standardized = X.copy()  # Sao chép dữ liệu để không làm thay đổi mảng X gốc.
    safe_std = np.where(std == 0, np.float32(1.0), std)  # Phòng trường hợp hàm được gọi với std chứa giá trị 0.
    X_standardized[:, :N_NUMERIC] = (X_standardized[:, :N_NUMERIC] - mean) / safe_std  # Chuẩn hoá 10 cột đầu theo công thức z-score.
    return X_standardized  # Trả mảng mới; 44 cột one-hot phía sau vẫn được giữ nguyên.


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42, processed_dir: str = "data/processed") -> dict:  # Định nghĩa hàm chuẩn bị toàn bộ tensor cho huấn luyện.
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)  # Nạp hai tập đã được chia sẵn từ ổ đĩa.
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)  # Tách validation chỉ từ train để không làm rò rỉ eval.
    mean, std = fit_standardizer(X_tr)  # Học thống kê chuẩn hoá chỉ trên phần train còn lại.
    X_tr = apply_standardizer(X_tr, mean, std)  # Chuẩn hoá đặc trưng của tập train.
    X_val = apply_standardizer(X_val, mean, std)  # Chuẩn hoá validation bằng đúng thống kê của train.
    X_eval = apply_standardizer(X_eval, mean, std)  # Chuẩn hoá eval bằng đúng thống kê của train, không học từ eval.
    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32, device=device)  # Chuyển X train thành tensor float32 trên thiết bị đã chọn.
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.int64, device=device)  # Chuyển nhãn train thành tensor int64 cho CrossEntropyLoss.
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)  # Chuyển X validation thành tensor float32 trên thiết bị đã chọn.
    y_val_tensor = torch.tensor(y_val, dtype=torch.int64, device=device)  # Chuyển nhãn validation thành tensor int64 trên thiết bị đã chọn.
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32, device=device)  # Chuyển X eval thành tensor float32 trên thiết bị đã chọn.
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.int64, device=device)  # Chuyển nhãn eval thành tensor int64 trên thiết bị đã chọn.
    majority_class = torch.bincount(y_tr_tensor, minlength=7).argmax()  # Tìm lớp xuất hiện nhiều nhất trong train.
    majority_accuracy = (y_val_tensor == majority_class).float().mean().item()  # Tính accuracy trên validation nếu luôn đoán lớp đa số.
    print(f"train={len(X_tr_tensor):,}, val={len(X_val_tensor):,}, eval={len(X_eval_tensor):,}")  # In số mẫu để kiểm tra nhanh việc tách dữ liệu.
    print(f"Majority-class baseline ({majority_class.item()}): val accuracy = {majority_accuracy:.4f}")  # In mốc baseline bằng ký tự ASCII để tương thích terminal Windows cũ.
    return {  # Gom tất cả kết quả vào một dict để các hàm huấn luyện dùng thuận tiện.
        "X_tr": X_tr_tensor,  # Tensor đặc trưng dùng để huấn luyện.
        "y_tr": y_tr_tensor,  # Tensor nhãn dùng để huấn luyện.
        "X_val": X_val_tensor,  # Tensor đặc trưng dùng để validation.
        "y_val": y_val_tensor,  # Tensor nhãn dùng để validation.
        "X_eval": X_eval_tensor,  # Tensor đặc trưng dùng cho dự đoán cuối cùng.
        "y_eval": y_eval_tensor,  # Tensor nhãn eval chỉ dùng để chấm điểm cuối.
        "eval_row_id": eval_row_id,  # Mã dòng NumPy để ghi tệp dự đoán theo đúng thứ tự.
    }  # Kết thúc dict dữ liệu đã chuẩn bị.


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):  # Định nghĩa generator trả từng batch dữ liệu.
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    assert batch_size > 0, "batch_size phải lớn hơn 0"  # Ngăn vòng lặp vô hạn hoặc chia batch không hợp lệ.
    assert len(X) == len(y), "X và y phải có cùng số mẫu"  # Đảm bảo mỗi đặc trưng luôn đi cùng đúng nhãn của nó.
    n_samples = len(X)  # Lưu số mẫu để dùng khi tạo chỉ số và cắt batch.
    if shuffle:  # Kiểm tra xem người gọi có muốn xáo dữ liệu ở epoch này không.
        indices = torch.randperm(n_samples, generator=generator, device=X.device)  # Tạo hoán vị ngẫu nhiên ngay trên cùng thiết bị với X.
    else:  # Đi vào nhánh này khi cần giữ nguyên thứ tự dữ liệu.
        indices = torch.arange(n_samples, device=X.device)  # Tạo dãy chỉ số 0, 1, ..., N - 1 không xáo trộn.
    for start in range(0, n_samples, batch_size):  # Duyệt điểm bắt đầu của từng batch, kể cả batch cuối nhỏ hơn batch_size.
        batch_indices = indices[start:start + batch_size]  # Lấy các chỉ số thuộc batch hiện tại.
        yield X[batch_indices], y[batch_indices]  # Trả một batch đặc trưng và nhãn tương ứng cho vòng lặp huấn luyện.
