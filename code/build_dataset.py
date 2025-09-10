import os
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scapy.all import PcapReader, IP, IPv6, TCP, UDP

# ---------- 保留原来的辅助函数 ----------
def _extract_label(fname: str) -> str:
    base = os.path.basename(fname)
    m = re.match(r"([^_]+)_.*\.(pcap|pcapng)$", base, re.IGNORECASE)
    return m.group(1) if m else os.path.splitext(base)[0]

def _biflow_key(ip_a, port_a, ip_b, port_b, proto: str):
    a, b = (ip_a, int(port_a)), (ip_b, int(port_b))
    ends = tuple(sorted((a, b)))
    return (ends[0], ends[1], proto)

def pcap_to_biflows(pcap_path: str, payload_only: bool = True):
    flows = {}
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            l3 = pkt.getlayer(IP) or pkt.getlayer(IPv6)
            if l3 is None: continue
            is_tcp = pkt.haslayer(TCP)
            is_udp = pkt.haslayer(UDP)
            if not (is_tcp or is_udp): continue

            if is_tcp:
                l4 = pkt.getlayer(TCP)
                proto = "TCP"
                payload_len = len(l4.payload) if l4 is not None else 0
            else:
                l4 = pkt.getlayer(UDP)
                proto = "UDP"
                payload_len = len(l4.payload) if l4 is not None else 0

            if payload_only and payload_len == 0: continue

            try:
                src, dst = l3.src, l3.dst
                sport, dport = int(l4.sport), int(l4.dport)
            except Exception:
                continue

            key = _biflow_key(src, sport, dst, dport, proto)
            ts = getattr(pkt, "time", None)
            size = payload_len if payload_only else len(bytes(pkt))
            now_dir = ((src, sport), (dst, dport))

            f = flows.get(key)
            if f is None:
                flows[key] = {"up": size, "down": 0, "first_ts": ts, "up_dir": now_dir}
            else:
                if f["up_dir"] == now_dir:
                    f["up"] += size
                else:
                    f["down"] += size
                if ts is not None and (f["first_ts"] is None or ts < f["first_ts"]):
                    f["first_ts"] = ts

    ordered = sorted(flows.values(), key=lambda x: (x["first_ts"] if x["first_ts"] is not None else float("inf")))
    features = [[int(f["up"]), int(f["down"])] for f in ordered]
    return features

# ---------- 多线程处理单个文件 ----------
def process_file(fpath, payload_only=True):
    name = os.path.basename(fpath)
    label = _extract_label(name)
    try:
        feats = pcap_to_biflows(fpath, payload_only=payload_only)
        total_up = sum(x[0] for x in feats)
        total_down = sum(x[1] for x in feats)
        return {
            "file": name,
            "label": label,
            "features": feats,
            "num_flows": len(feats),
            "total_up": total_up,
            "total_down": total_down,
        }
    except Exception as e:
        print(f"[WARN] 解析失败: {name} -> {e}")
        return None

# ---------- 多线程构建数据集 ----------
def build_dataset_from_folder_multithread(folder: str,
                                           payload_only: bool = True,
                                           exts=(".pcap", ".pcapng"),
                                           max_workers: int = 8) -> pd.DataFrame:
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    rows = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f, payload_only): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            result = fut.result()
            if result is not None:
                rows.append(result)
    print(len(files), '个文件已完成。')
    df = pd.DataFrame(rows, columns=["file", "label", "features", "num_flows", "total_up", "total_down"])
    return df


if __name__ == '__main__':
    # ---------- 示例 ----------
    folder = r"F:\dataset\baidu20\20250823"
    print(folder)
    df = build_dataset_from_folder_multithread(folder, payload_only=True, max_workers=8)
    df.to_pickle("baidu20_0823.pkl")
