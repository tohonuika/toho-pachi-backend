from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dataclasses import dataclass
import random
import math

app = FastAPI(title="Pachi Monte Carlo Simulator")


# ======== シミュレーション用データクラス ========

@dataclass
class MachineBase:
    hit_prob: float         # 1回転あたりの大当たり確率 (例: 1/319)
    continue_prob: float    # 継続率 (0〜1)
    cost_per_spin: float    # 1回転あたりの投資


@dataclass
class RowData:
    day_index: int          # 1=7日前, ..., 8=当日
    machine_no: str         # 台番号
    total_spins: int        # その日の総回転数
    total_hits: int         # 実績の大当り回数
    net_diff: float         # 実績の差玉
    one_set_max: float      # 実績の1セット最大出玉
    max_chain_len: int      # 実績の最大連チャン数（今は未使用）


@dataclass
class SimResult:
    win_rate: float
    avg_profit: float
    avg_max_payout: float
    avg_max_invest: float
    max_payout_p90: float
    max_invest_p90: float
    explosion_rate: float
    explosion_spins_avg: float
    explosion_invest_avg: float


# ======== API 入力モデル ========

class MachineInput(BaseModel):
    day_index: int
    machine_no: str
    total_spins: Optional[int] = 0
    total_hits: Optional[int] = 0
    net_diff: Optional[float] = 0.0
    one_set_max: Optional[float] = 0.0
    max_chain_len: Optional[int] = 0


class SimulateRequest(BaseModel):
    hit_prob_den: float          # 1/X の X (例: 319)
    continue_prob: float         # 継続率(％, 例: 81)
    cost_per_spin: float         # 1回転あたりの投資
    rows: List[MachineInput]     # 1行=1台


# ======== ヘルパー関数 ========

def percentile(values, q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return float(values[int(k)])
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def estimate_avg_payout_per_hit(rows: List[RowData], base: MachineBase) -> float:
    """
    実績データから「大当り1回あたりの平均出玉」を推定。
      total_payout = net_diff + total_invest
      avg_payout_per_hit = total_payout / total_hits
    """
    vals = []
    for r in rows:
        if r.total_hits is None or r.total_hits <= 0 or r.total_spins <= 0:
            continue
        total_invest = base.cost_per_spin * r.total_spins
        total_payout = r.net_diff + total_invest
        vals.append(total_payout / r.total_hits)

    if not vals:
        # データが全くないときの保険
        return 1000.0

    return sum(vals) / len(vals)


def estimate_one_set_max(rows: List[RowData], avg_payout: float) -> float:
    """
    1セット最大出玉の実績からざっくり上限を推定。
    """
    vals = [r.one_set_max for r in rows if r.one_set_max and r.one_set_max > 0]
    if vals:
        return sum(vals) / len(vals)
    # 実績が無ければ平均の3倍くらいを上限とみなす
    return avg_payout * 3.0


def compute_explosion_threshold(avg_payout: float, one_set_max_est: float) -> float:
    """
    「爆発」とみなす1回の連チャン出玉の閾値。
      ・1セット最大出玉の70％
      ・平均出玉の3倍
    のうち大きい方。
    """
    return max(one_set_max_est * 0.7, avg_payout * 3.0)


def sample_geometric_miss_count(p: float) -> int:
    """
    次の大当りまでのハマり回数を幾何分布からサンプル。
    戻り値は 0,1,2,... の「ハズレ回数」。
    """
    if p <= 0.0:
        return 10 ** 9
    if p >= 1.0:
        return 0
    u = random.random()
    return int(math.log(1 - u) / math.log(1 - p))


def sample_payout(avg_payout: float, one_set_max_est: float) -> float:
    """
    RUSH1セットあたりの出玉を「平均±ちょいバラつき」でサンプル。
    正規分布 + 0〜上限でクリップ。
    """
    sigma = avg_payout * 0.5
    x = random.gauss(avg_payout, sigma)
    if x < 0:
        x = 0.0
    if x > one_set_max_est:
        x = one_set_max_est
    return x


def simulate_one_day_fast(
    base: MachineBase,
    total_spins: int,
    avg_payout_per_hit: float,
    one_set_max_est: float,
    explosion_threshold: float,
) -> dict:
    """
    1台ぶん（その日の総回転数）のシミュレーション。
      ・ハマり区間は幾何分布で一気にジャンプ
      ・RUSH中だけ細かく処理
    """
    bankroll = 0.0
    max_bankroll = 0.0
    min_bankroll = 0.0
    spins_done = 0

    explosion_spins = 0
    explosion_invest = 0.0

    while spins_done < total_spins:
        # ハズレ区間
        miss_count = sample_geometric_miss_count(base.hit_prob)

        if spins_done + miss_count >= total_spins:
            remaining = total_spins - spins_done
            bankroll -= base.cost_per_spin * remaining
            spins_done = total_spins
            min_bankroll = min(min_bankroll, bankroll)
            break
        else:
            bankroll -= base.cost_per_spin * miss_count
            spins_done += miss_count
            min_bankroll = min(min_bankroll, bankroll)

        if spins_done >= total_spins:
            break

        # 当たりを引いた回転
        bankroll -= base.cost_per_spin
        spins_done += 1
        min_bankroll = min(min_bankroll, bankroll)
        hit_spin_index = spins_done

        # RUSH / ST 連チャン
        chain_payout = 0.0
        while True:
            payout = sample_payout(avg_payout_per_hit, one_set_max_est)
            bankroll += payout
            chain_payout += payout

            if bankroll > max_bankroll:
                max_bankroll = bankroll

            # 初めて爆発条件を満たしたら記録
            if explosion_spins == 0 and chain_payout >= explosion_threshold:
                explosion_spins = hit_spin_index
                explosion_invest = base.cost_per_spin * hit_spin_index

            # 継続抽選
            if random.random() >= base.continue_prob:
                break

    return {
        "final_profit": bankroll,
        "max_payout": max_bankroll,
        "max_invest": -min_bankroll,
        "explosion_spins": explosion_spins,
        "explosion_invest": explosion_invest,
    }


def run_simulation_for_row(
    base: MachineBase,
    row: RowData,
    avg_payout_per_hit: float,
    one_set_max_est: float,
    explosion_threshold: float,
    n_sim: int,
) -> SimResult:
    finals = []
    max_payouts = []
    max_invests = []
    explosion_spins_vals = []
    explosion_invest_vals = []

    for _ in range(n_sim):
        r = simulate_one_day_fast(
            base,
            row.total_spins,
            avg_payout_per_hit,
            one_set_max_est,
            explosion_threshold,
        )
        finals.append(r["final_profit"])
        max_payouts.append(r["max_payout"])
        max_invests.append(r["max_invest"])

        if r["explosion_spins"] > 0:
            explosion_spins_vals.append(r["explosion_spins"])
            explosion_invest_vals.append(r["explosion_invest"])

    n = n_sim if n_sim > 0 else 1
    win_rate = sum(1 for x in finals if x > 0) / n
    avg_profit = sum(finals) / n
    avg_max_payout = sum(max_payouts) / n
    avg_max_invest = sum(max_invests) / n

    explosion_rate = len(explosion_spins_vals) / n if n > 0 else 0.0
    explosion_spins_avg = (
        sum(explosion_spins_vals) / len(explosion_spins_vals)
        if explosion_spins_vals else 0.0
    )
    explosion_invest_avg = (
        sum(explosion_invest_vals) / len(explosion_invest_vals)
        if explosion_invest_vals else 0.0
    )

    return SimResult(
        win_rate=win_rate,
        avg_profit=avg_profit,
        avg_max_payout=avg_max_payout,
        avg_max_invest=avg_max_invest,
        max_payout_p90=percentile(max_payouts, 90),
        max_invest_p90=percentile(max_invests, 90),
        explosion_rate=explosion_rate,
        explosion_spins_avg=explosion_spins_avg,
        explosion_invest_avg=explosion_invest_avg,
    )


def analyze_machines(
    base: MachineBase,
    rows: List[RowData],
    n_sim: int,
):
    """
    rows: 1行=1台（day_index は 1=7日前, ..., 8=当日）。
    - 平均出玉 / 爆発条件 は全 rows から推定。
    - Monte Carlo 対象は:
        * 当日(day_index=8) で total_spins>0 の台があれば、その台だけ
        * それが無ければ 7日前〜1日前(day_index=1..7) の台
    """
    # 平均出玉・爆発条件を推定
    avg_payout = estimate_avg_payout_per_hit(rows, base)
    one_set_max_est = estimate_one_set_max(rows, avg_payout)
    explosion_threshold = compute_explosion_threshold(avg_payout, one_set_max_est)

    # どの台を Monte Carlo するか決める
    has_today = any(r.day_index == 8 and r.total_spins > 0 for r in rows)
    if has_today:
        target_rows = [r for r in rows if r.day_index == 8 and r.total_spins > 0]
    else:
        target_rows = [r for r in rows if 1 <= r.day_index <= 7 and r.total_spins > 0]

    results = []
    for row in target_rows:
        stats = run_simulation_for_row(
            base,
            row,
            avg_payout,
            one_set_max_est,
            explosion_threshold,
            n_sim,
        )
        results.append({
            "day_index": row.day_index,
            "machine_no": row.machine_no,
            "total_spins": row.total_spins,
            "win_rate": stats.win_rate,
            "avg_profit": stats.avg_profit,
            "max_invest_90": stats.max_invest_p90,
            "max_payout_90": stats.max_payout_p90,
            "explosion_rate": stats.explosion_rate,
            "explosion_spins_avg": stats.explosion_spins_avg,
            "explosion_invest_avg": stats.explosion_invest_avg,
        })

    # 勝率順で TOP5
    results.sort(key=lambda x: x["win_rate"], reverse=True)
    return {
        "avg_payout_per_hit": avg_payout,
        "one_set_max_est": one_set_max_est,
        "explosion_threshold": explosion_threshold,
        "top5": results[:5],
    }


# ======== API エンドポイント ========

@app.post("/simulate")
def simulate(req: SimulateRequest):
    # モンテカルロ試行回数（サーバ側で固定）
    DEFAULT_N_SIM = 8000

    if req.hit_prob_den <= 0:
        raise HTTPException(status_code=400, detail="hit_prob_den must be > 0")

    hit_prob = 1.0 / req.hit_prob_den
    cont_prob = max(0.0, min(1.0, req.continue_prob / 100.0))

    if req.cost_per_spin <= 0:
        raise HTTPException(status_code=400, detail="cost_per_spin must be > 0")

    if not req.rows:
        raise HTTPException(status_code=400, detail="rows must not be empty")

    base = MachineBase(
        hit_prob=hit_prob,
        continue_prob=cont_prob,
        cost_per_spin=req.cost_per_spin,
    )

    rows_data: List[RowData] = []
    for r in req.rows:
        machine_no = (r.machine_no or "").strip()
        if not machine_no:
            continue

        day_idx = int(r.day_index or 1)
        if day_idx < 1:
            day_idx = 1
        if day_idx > 8:
            day_idx = 8

        td = int(r.total_spins or 0)
        th = int(r.total_hits or 0)
        nd = float(r.net_diff or 0.0)
        osm = float(r.one_set_max or 0.0)
        mcl = int(r.max_chain_len or 0)

        rows_data.append(RowData(
            day_index=day_idx,
            machine_no=machine_no,
            total_spins=td,
            total_hits=th,
            net_diff=nd,
            one_set_max=osm,
            max_chain_len=mcl,
        ))

    if not any(r.total_spins > 0 for r in rows_data):
        raise HTTPException(status_code=400, detail="At least one row must have total_spins > 0")

    summary = analyze_machines(base, rows_data, n_sim=DEFAULT_N_SIM)

    return {
        "hit_prob": hit_prob,
        "continue_prob": cont_prob,
        "cost_per_spin": base.cost_per_spin,
        "n_sim": DEFAULT_N_SIM,
        **summary,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
