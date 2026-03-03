import json
import math
from pathlib import Path
from collections import Counter

# ------------------ 基础函数 ------------------
def edge_length(junctions, edge):
    """计算单条线段的长度"""
    i, j = edge
    x1, y1 = junctions[i]
    x2, y2 = junctions[j]
    return math.hypot(x2 - x1, y2 - y1)


def compute_lengths(json_path):
    """统计平均长度与区间分布"""
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_lengths = []
    for img in data:
        junctions = img.get("junctions", [])
        edges = img.get("edges_positive", [])
        for e in edges:
            if e[0] < len(junctions) and e[1] < len(junctions):
                length = edge_length(junctions, e)
                all_lengths.append(length)

    if not all_lengths:
        print(f"⚠️ {json_path.name} 中未找到有效线段。")
        return 0, 0, {}

    # 平均值统计
    avg_length = sum(all_lengths) / len(all_lengths)
    min_len, max_len = min(all_lengths), max(all_lengths)

    print(f"📊 {json_path.name} 统计：")
    print(f"  - 导线数量: {len(all_lengths)}")
    print(f"  - 平均长度: {avg_length:.2f} px")
    print(f"  - 最短导线: {min_len:.2f} px")
    print(f"  - 最长导线: {max_len:.2f} px")

    # ------------------ 区间统计 ------------------
    bins = list(range(0, 401, 10))  # 0–400，每 10px 一段
    bin_counter = Counter()

    for l in all_lengths:
        # 归类到 0–10, 10–20 ... 390–400 区间
        bin_index = int(min(l, 400) // 10) * 10
        bin_counter[bin_index] += 1

    print("\n📈 导线长度区间分布（单位: px）:")
    for b in bins:
        count = bin_counter.get(b, 0)
        print(f"  {b:3d}–{b+10:3d} px : {count:5d}")

    print()
    return len(all_lengths), avg_length, bin_counter


# ------------------ 主执行入口 ------------------
if __name__ == "__main__":
    base = "/root/autodl-tmp/HAWP/data/data_hawp_last/json_converted"

    train_file = f"{base}/train_hawp_style.json"
    val_file = f"{base}/val_hawp_style.json"

    n1, avg1, bins1 = compute_lengths(train_file)
    n2, avg2, bins2 = compute_lengths(val_file)

    if (n1 + n2) > 0:
        total_avg = (avg1 * n1 + avg2 * n2) / (n1 + n2)
        print("📊 综合统计结果：")
        print(f"  - 总导线数量: {n1 + n2}")
        print(f"  - 平均导线长度: {total_avg:.2f} px")

        # 合并区间计数
        total_bins = Counter()
        total_bins.update(bins1)
        total_bins.update(bins2)

        print("\n📈 综合长度区间分布（0–400 px）:")
        for b in range(0, 401, 10):
            print(f"  {b:3d}–{b+10:3d} px : {total_bins.get(b, 0):5d}")





    