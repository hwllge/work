"""
train_rps_mlp.py

MediaPipe 손 관절 좌표  →  인접 관절 코사인 각도  →  MLP (PyTorch)

라벨:  0=scissors  1=rock  2=paper

관절 각도 계산 방법:
  각 손가락 체인 내 중간 관절(i)에서,
    v1 = landmark[i]   - landmark[i-1]   (이전 관절 → 현재 관절 벡터)
    v2 = landmark[i+1] - landmark[i]     (현재 관절 → 다음 관절 벡터)
    cosine = dot(v1, v2) / (|v1| * |v2|)
  → 손가락을 폈을 때 ≈ +1,  구부렸을 때 감소

손가락 체인(각 체인당 3개 각도) × 5개 손가락 = 15 feature:
  엄지: [0,1,2,3,4]  → 관절 1,2,3
  검지: [0,5,6,7,8]  → 관절 5,6,7
  중지: [0,9,10,11,12] → 관절 9,10,11
  약지: [0,13,14,15,16]→ 관절 13,14,15
  소지: [0,17,18,19,20]→ 관절 17,18,19

0번 관절(손목) 4개 추가 feature (총 19):
  벡터 기준: landmark[a] - landmark[0]
  cos(0→1, 0→5), cos(0→1, 0→9), cos(0→1, 0→13), cos(0→1, 0→17)
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# ─────────────────────────────────────────────
# 관절 각도 설정
# ─────────────────────────────────────────────
FINGER_CHAINS = [
    [0,  1,  2,  3,  4],   # 엄지
    [0,  5,  6,  7,  8],   # 검지
    [0,  9, 10, 11, 12],   # 중지
    [0, 13, 14, 15, 16],   # 약지
    [0, 17, 18, 19, 20],   # 소지
]
# 0번 관절(손목)에서 0→1 벡터와 0→finger_base 벡터 사이 코사인 4개
WRIST_PAIRS = [(1, 5), (1, 9), (1, 13), (1, 17)]
NUM_FEATURES = sum(len(c) - 2 for c in FINGER_CHAINS) + len(WRIST_PAIRS)  # 19
LABEL_MAP    = {0: "scissors", 1: "rock", 2: "paper"}


# ─────────────────────────────────────────────
# MediaPipe HandLandmarker (Tasks API)
# ─────────────────────────────────────────────
HAND_LANDMARKER_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_PATH = "hand_landmarker.task"


def get_hand_landmarker(
    model_path: str = HAND_LANDMARKER_PATH,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.3,
) -> mp_vision.HandLandmarker:
    """hand_landmarker.task 파일이 없으면 다운로드 후 HandLandmarker 반환."""
    if not Path(model_path).exists():
        print(f"hand_landmarker.task 다운로드 중 → {model_path}")
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
# Feature 추출
# ─────────────────────────────────────────────
def cosine_angle(p_prev: np.ndarray, p_cur: np.ndarray, p_next: np.ndarray) -> float:
    """p_cur 관절에서 인접 두 벡터 사이의 코사인값 반환."""
    v1 = p_cur  - p_prev
    v2 = p_next - p_cur
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-7:
        return 1.0          # 좌표가 동일 → 직선으로 처리
    return float(np.dot(v1, v2) / denom)


def wrist_cosine(origin: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> float:
    """origin에서 p_a, p_b 방향 벡터 사이의 코사인값 반환."""
    v1 = p_a - origin
    v2 = p_b - origin
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-7:
        return 1.0
    return float(np.dot(v1, v2) / denom)


def landmarks_to_features(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks : (21, 3)  – MediaPipe 정규화 좌표 (x, y, z)
    반환      : (19,)    – 손가락 체인 15개 + 손목 4개 코사인값
    """
    features = []
    # 손가락 체인 내 중간 관절 코사인 (15개)
    for chain in FINGER_CHAINS:
        for i in range(1, len(chain) - 1):
            features.append(
                cosine_angle(
                    landmarks[chain[i - 1]],
                    landmarks[chain[i]],
                    landmarks[chain[i + 1]],
                )
            )
    # 0번 관절(손목) 기준 코사인 (4개)
    for a, b in WRIST_PAIRS:
        features.append(wrist_cosine(landmarks[0], landmarks[a], landmarks[b]))
    return np.array(features, dtype=np.float32)


def process_image(img_path: Path, landmarker: mp_vision.HandLandmarker) -> "np.ndarray | None":
    """이미지 경로 → 관절 각도 feature. 손 미검출 시 None."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = np.array(
        [[p.x, p.y, p.z] for p in result.hand_landmarks[0]],
        dtype=np.float32,
    )
    return landmarks_to_features(lm)


def build_feature_cache(
    data_root: str,
    cache_path: Path,
    hand_model: str = HAND_LANDMARKER_PATH,
):
    """
    data_root/<0,1,2>/ 이미지에서 feature를 추출해 .npz 캐시 저장.
    반환: (X: ndarray [N,19], y: ndarray [N])
    """
    landmarker = get_hand_landmarker(model_path=hand_model)

    X_list, y_list, skipped = [], [], 0

    for label in [0, 1, 2]:
        label_dir = Path(data_root) / str(label)
        if not label_dir.exists():
            print(f"[WARN] {label_dir} 없음, 건너뜀")
            continue
        imgs = sorted(
            p for p in label_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"  class {label} ({LABEL_MAP[label]}): {len(imgs)}장")
        for img_path in tqdm(imgs, desc=f"  class {label}", leave=False):
            feats = process_image(img_path, landmarker)
            if feats is None:
                skipped += 1
                continue
            X_list.append(feats)
            y_list.append(label)

    landmarker.close()
    print(f"  추출 완료: {len(X_list)}개 샘플 | {skipped}개 건너뜀 (손 미검출)")

    if len(X_list) == 0:
        found = [str(p) for p in Path(data_root).glob("*")] if Path(data_root).exists() else []
        raise RuntimeError(
            f"데이터셋에서 샘플을 하나도 추출하지 못했습니다.\n"
            f"  data_root: {data_root}\n"
            f"  실제 존재하는 항목: {found if found else '(폴더 자체가 없음)'}\n"
            f"  0/1/2 하위 폴더 이름을 확인하세요."
        )

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    np.savez(str(cache_path), X=X, y=y)
    print(f"  캐시 저장 → {cache_path}")
    return X, y


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class JointAngleDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# MLP 모델
# ─────────────────────────────────────────────
class RPSMLP(nn.Module):
    """
    small MLP:  [input] → (Linear → BN → ReLU → Dropout) × N → [3]

    기본 구조: 15 → 64 → 32 → 3
    """
    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        hidden_dims: tuple = (64, 32),
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            out  = model(X_b)
            loss = criterion(out, y_b)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_b)
            correct    += (out.argmax(1) == y_b).sum().item()
            total      += len(y_b)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RPS joint-angle MLP trainer")
    parser.add_argument(
        "--data",
        default="/content/drive/MyDrive/files/RPS_Dataset/RPS_Dataset",
        help="데이터셋 루트 (하위에 0/1/2 폴더)",
    )
    parser.add_argument("--cache",        default="joint_angles.npz",                    help="feature 캐시 경로")
    parser.add_argument("--out-model",    default="/content/drive/MyDrive/files/rps_mlp.pt", help="저장할 모델 파일")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--val-ratio",    type=float, default=0.15)
    parser.add_argument("--test-ratio",   type=float, default=0.15)
    parser.add_argument("--hidden",       nargs="+",  type=int, default=[64, 32],
                        help="은닉층 크기 (ex: --hidden 128 64 32)")
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="캐시가 있어도 feature를 다시 추출")
    parser.add_argument("--patience",     type=int,   default=15,
                        help="Early stopping patience (val acc 기준, 0=비활성화)")
    parser.add_argument("--hand-model",   default=HAND_LANDMARKER_PATH,
                        help="hand_landmarker.task 경로 (없으면 자동 다운로드)")
    args, _ = parser.parse_known_args()  # Jupyter/Colab 커널 인수 무시

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    print(f"device: {device}  |  feature 수: {NUM_FEATURES}")

    # ── feature 로드 / 추출 ──
    cache = Path(args.cache)
    if cache.exists() and not args.rebuild_cache:
        print(f"캐시 로드: {cache}")
        d    = np.load(str(cache))
        X, y = d["X"], d["y"]
    else:
        print("이미지에서 관절 각도 feature 추출 중 ...")
        X, y = build_feature_cache(args.data, cache, hand_model=args.hand_model)

    print(f"데이터셋: {len(X)}샘플  feature={X.shape[1]}  클래스={sorted(set(y.tolist()))}")

    # ── 데이터 분할 ──
    dataset = JointAngleDataset(X, y)
    n       = len(dataset)
    n_val   = int(n * args.val_ratio)
    n_test  = int(n * args.test_ratio)
    n_train = n - n_val - n_test
    gen     = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)
    print(f"분할  train={n_train}  val={n_val}  test={n_test}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True, pin_memory=use_cuda, num_workers=2 if use_cuda else 0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, pin_memory=use_cuda, num_workers=2 if use_cuda else 0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, pin_memory=use_cuda, num_workers=2 if use_cuda else 0)

    # ── 모델 / 옵티마이저 ──
    model     = RPSMLP(input_dim=X.shape[1], hidden_dims=args.hidden, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n모델 구조: {model}\n")

    # ── 학습 ──
    best_val_acc = 0.0
    no_improve   = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss,  tr_acc  = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), args.out_model)
            flag = "  ← best"
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or flag:
            print(
                f"epoch {epoch:3d}/{args.epochs}  "
                f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
                f"val loss={val_loss:.4f} acc={val_acc:.3f}{flag}"
            )

        if args.patience > 0 and no_improve >= args.patience:
            print(f"\nEarly stopping: val acc {args.patience}epoch 동안 개선 없음 (epoch {epoch})")
            break

    # ── 테스트 ──
    print(f"\n최고 val acc: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(args.out_model, map_location=device, weights_only=True))
    _, test_acc = run_epoch(model, test_loader, criterion, None, device, train=False)
    print(f"test acc:     {test_acc:.4f}")
    print(f"모델 저장:    {args.out_model}")


if __name__ == "__main__":
    main()
