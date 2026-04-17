import socket
import json
import torch
import time
import argparse
import os
import glob
import urllib.request
import urllib.error
from collections import deque
from models.model import KinematicFusionModel
from processor.filter import OneEuroFilter
from processor.preprocessor import preprocess_five_sensors, parse_sse_payload

# ==============================
# デバイスID（左足・右足・左腕・右腕・腰）
# ==============================
LEFT_FOOT_DN  = "3030F9284F54"
RIGHT_FOOT_DN = "3030F92685D4"
LEFT_ARM_DN   = "B8F862C6FDD4"
RIGHT_ARM_DN  = "F0F5BD5DAED0"
BACK_C_DN     = "50787D1ADCDC"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream_url", type=str, default="http://163.143.136.103:5001/stream", help="HTTP Stream URL")
    parser.add_argument("--send_ip",   type=str, default="127.0.0.1", help="UDP Sending IP")
    parser.add_argument("--send_port", type=int, default=5006,         help="UDP Sending Port")
    parser.add_argument("--weights",   type=str, default="",           help="Path to model weights (.pth)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==============================
    # 1. モデルとフィルタの初期化
    # ==============================
    NUM_JOINTS = 24
    model = KinematicFusionModel(
        foot_features=70, imu_sensors=5, imu_channels=9, num_joints=NUM_JOINTS
    ).to(device)

    # 最新の重みを自動選択
    weight_files = sorted(glob.glob("weights/*.pth"))
    TARGET_WEIGHT_PATH = weight_files[-1] if weight_files else ""
    weight_to_load = args.weights if args.weights else TARGET_WEIGHT_PATH

    if weight_to_load:
        if os.path.exists(weight_to_load):
            try:
                model.load_state_dict(torch.load(weight_to_load, map_location=device, weights_only=True))
                print(f"Loaded weights from {weight_to_load}")
            except Exception as e:
                print(f"Failed to load weights: {e}")
                raise RuntimeError("モデルと重みのアーキテクチャが一致しません。") from e
        else:
            print(f"Warning: Weight file '{weight_to_load}' not found.")

    model.eval()
    model.set_stateful(False)  # リアルタイム推論モード（ステートレス・スライディングウィンドウ方式）

    # 施策D: OneEuroFilter パラメータ最適化
    # mincutoff=0.5: 小さくするほど遅い動きのスムーシングが強化される
    # beta=0.003: 大きくするほど速い動きに素早く追従する
    # dcutoff=1.0: 速度のカットオフ周波数（デフォルトのままで問題なし）
    euro_filter = OneEuroFilter(mincutoff=0.5, beta=0.003, dcutoff=1.0)

    # ==============================
    # 2. 通信の初期化
    # ==============================
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Connecting to HTTP Stream at {args.stream_url}")
    print(f"Output will be sent to {args.send_ip}:{args.send_port}")
    print(f"Left foot DN : {LEFT_FOOT_DN}")
    print(f"Right foot DN: {RIGHT_FOOT_DN}")
    print(f"Left arm DN  : {LEFT_ARM_DN}")
    print(f"Right arm DN : {RIGHT_ARM_DN}")
    print(f"Back C DN    : {BACK_C_DN}")
    print("Waiting for data from all 5 sensors...")

    # ==============================
    # 3. センサーバッファ
    # ==============================
    # 各センサーの最新フレームデータをキャッシュし、全て揃ったら推論する
    buffers = {
        LEFT_FOOT_DN: None,
        RIGHT_FOOT_DN: None,
        LEFT_ARM_DN: None,
        RIGHT_ARM_DN: None,
        BACK_C_DN: None
    }

    # スライディングウィンドウ用のキュー
    SEQ_LEN = 50
    sliding_foot = deque(maxlen=SEQ_LEN)
    sliding_imu = deque(maxlen=SEQ_LEN)

    # ==============================
    # 4. リアルタイム推論ループ
    # ==============================
    try:
        req = urllib.request.Request(args.stream_url)
        with urllib.request.urlopen(req) as response:
            for line in response:
                raw_str = line.decode('utf-8').strip()
                if not raw_str:
                    continue

                parsed = parse_sse_payload(raw_str)
                if not parsed:
                    continue

                # ストリームのルートに直接 dn と payload がある場合と
                # payload 内に dn がある場合の両方に対応
                dn = parsed.get("dn", "")
                payload_data = parsed.get("payload", {})
                if not dn and payload_data:
                    dn = payload_data.get("dn", "")

                p_data = payload_data.get("p",    [])
                acc    = payload_data.get("acc",   [])
                gyro   = payload_data.get("gyro",  [])
                mag    = payload_data.get("mag",   [0.0, 0.0, 0.0])

                # データの形状チェック
                if dn in [LEFT_FOOT_DN, RIGHT_FOOT_DN]:
                    if len(p_data) != 35 or len(acc) != 3 or len(gyro) != 3:
                        continue
                else:
                    if len(acc) != 3 or len(gyro) != 3:
                        continue

                frame_data = {"p": p_data, "acc": acc, "gyro": gyro, "mag": mag}

                # デバイスごとのバッファに保存
                if dn in buffers:
                    buffers[dn] = frame_data
                else:
                    # 未知のデバイスはスキップ
                    continue

                # 全てのデータが揃っていなければ次のフレームへ
                if any(v is None for v in buffers.values()):
                    print(f"[{dn}] データ受信中. 他のセンサーを待機...")
                    continue

                # ==============================
                # 5. 前処理（5箇所のデータ使用）
                # ==============================
                foot_tensor, imu_tensor = preprocess_five_sensors(
                    p_l     = buffers[LEFT_FOOT_DN]["p"],
                    acc_l   = buffers[LEFT_FOOT_DN]["acc"],
                    gyro_l  = buffers[LEFT_FOOT_DN]["gyro"],
                    mag_l   = buffers[LEFT_FOOT_DN]["mag"],
                    p_r     = buffers[RIGHT_FOOT_DN]["p"],
                    acc_r   = buffers[RIGHT_FOOT_DN]["acc"],
                    gyro_r  = buffers[RIGHT_FOOT_DN]["gyro"],
                    mag_r   = buffers[RIGHT_FOOT_DN]["mag"],
                    acc_al  = buffers[LEFT_ARM_DN]["acc"],
                    gyro_al = buffers[LEFT_ARM_DN]["gyro"],
                    mag_al  = buffers[LEFT_ARM_DN]["mag"],
                    acc_ar  = buffers[RIGHT_ARM_DN]["acc"],
                    gyro_ar = buffers[RIGHT_ARM_DN]["gyro"],
                    mag_ar  = buffers[RIGHT_ARM_DN]["mag"],
                    acc_bc  = buffers[BACK_C_DN]["acc"],
                    gyro_bc = buffers[BACK_C_DN]["gyro"],
                    mag_bc  = buffers[BACK_C_DN]["mag"],
                    device  = device
                )

                # ==============================
                # 6. 推論（スライディングウィンドウ）
                # ==============================
                sliding_foot.append(foot_tensor.squeeze(0).squeeze(0))
                sliding_imu.append(imu_tensor.squeeze(0).squeeze(0))

                if len(sliding_foot) < SEQ_LEN:
                    # 50フレーム溜まるまでは待機
                    continue

                start_time = time.time()
                with torch.no_grad():
                    foot_seq = torch.stack(list(sliding_foot)).unsqueeze(0)
                    imu_seq = torch.stack(list(sliding_imu)).unsqueeze(0)
                    
                    out_pos = model(foot_seq, imu_seq)
                    out_pos_last = out_pos[:, -1:, :, :]
                    
                    out_pos_filtered = euro_filter(start_time, out_pos_last)

                latency_ms = (time.time() - start_time) * 1000

                # ==============================
                # 7. 後処理とデータ送信
                # ==============================
                # (B=1, Seq=1, Joints=24, 3) -> (24, 3) のリスト
                pos_list = out_pos_filtered.squeeze().cpu().numpy().tolist()

                output_msg = {
                    "ts": time.time(),
                    "latency_ms": round(latency_ms, 2),
                    "pose_positions": pos_list
                }

                sock_send.sendto(
                    json.dumps(output_msg).encode('utf-8'),
                    (args.send_ip, args.send_port)
                )

                print(f"Processed frame (5 sensors). Latency: {latency_ms:.2f} ms")

    except KeyboardInterrupt:
        print("\nStopped real-time inference.")
    except urllib.error.URLError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during processing: {e}")
    finally:
        sock_send.close()

if __name__ == "__main__":
    main()
