"""最小 greedy speculative decoding 模拟器。

这个文件不依赖 PyTorch/Numpy，专门用于理解算法流程：

    Draft block -> Target verify -> accept prefix -> bonus token
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifyResult:
    accepted: int
    next_token: str
    emitted: list[str]


def argmax(probabilities: dict[str, float]) -> str:
    """返回概率最大的 token；同概率时按 token 名字稳定选择。"""
    return max(probabilities, key=lambda token: (probabilities[token], token))


def verify_greedy(
    draft_tokens: list[str],
    target_rows: list[dict[str, float]],
) -> VerifyResult:
    """使用 Target 的 logits 验证 Draft，并返回最长接受前缀和 bonus token。

    target_rows[i] 对应 Draft token[i] 所在位置的 Target 分布。
    target_rows[len(draft_tokens)] 是全部 Draft 接受后的 bonus 位置。
    """
    accepted = 0

    for index, draft_token in enumerate(draft_tokens):
        target_token = argmax(target_rows[index])
        if target_token != draft_token:
            break
        accepted += 1

    # accepted 既是第一个不匹配位置，也是 bonus token 所在位置。
    next_token = argmax(target_rows[accepted])
    emitted = draft_tokens[:accepted] + [next_token]
    return VerifyResult(accepted, next_token, emitted)


def expected_acceptance_length(alpha: float, draft_length: int) -> float:
    """假设每个位置独立且接受率都是 alpha，计算 E[A]。"""
    return sum(alpha**i for i in range(1, draft_length + 1))


def estimate_speedup(
    alpha: float,
    draft_length: int,
    target_single_cost_ms: float,
    draft_cost_ms: float,
    verify_cost_ms: float,
    cache_cost_ms: float,
) -> float:
    """使用粗略成本模型估计 speculative speedup。"""
    expected_accepted = expected_acceptance_length(alpha, draft_length)
    expected_emitted = 1.0 + expected_accepted
    speculative_round_cost = draft_cost_ms + verify_cost_ms + cache_cost_ms
    speculative_tps = expected_emitted / speculative_round_cost
    normal_tps = 1.0 / target_single_cost_ms
    return speculative_tps / normal_tps


def demo() -> None:
    # Draft 预测 5 个 token。
    draft = ["A", "B", "C", "D", "E"]

    # Target 对前 5 个位置的判断是 A、B、X、Y、Z；最后一行是 bonus 位置。
    target_rows = [
        {"A": 0.8, "X": 0.1, "B": 0.1},
        {"B": 0.7, "X": 0.2, "C": 0.1},
        {"X": 0.9, "C": 0.05, "D": 0.05},
        {"Y": 0.8, "D": 0.1, "E": 0.1},
        {"Z": 0.8, "E": 0.1, "X": 0.1},
        {"Q": 0.8, "Z": 0.1, "X": 0.1},
    ]

    result = verify_greedy(draft, target_rows)
    print("draft:", draft)
    print("accepted draft tokens:", result.accepted)
    print("next token:", result.next_token)
    print("emitted:", result.emitted)

    alpha = 0.8
    k = 5
    expected_a = expected_acceptance_length(alpha, k)
    print("expected acceptance length:", round(expected_a, 4))
    print(
        "estimated speedup:",
        round(
            estimate_speedup(
                alpha=alpha,
                draft_length=k,
                target_single_cost_ms=10.0,
                draft_cost_ms=2.0,
                verify_cost_ms=12.0,
                cache_cost_ms=1.0,
            ),
            4,
        ),
    )


if __name__ == "__main__":
    demo()

