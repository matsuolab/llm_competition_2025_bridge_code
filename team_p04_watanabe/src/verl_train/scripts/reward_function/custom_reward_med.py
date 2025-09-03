import re, math
from typing import Optional, Tuple

_HASH_RE = re.compile(r"^\s*####\s*(.+?)\s*$", flags=re.MULTILINE)

# ── Valid/canonical Roman up to 3999 (we'll clamp later) ──
_ROMAN_CANONICAL_RE = re.compile(
    r"^M{0,3}(CM|CD|D?C{0,3})"
    r"(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
_ROMAN_VAL = {
    "M":1000, "CM":900, "D":500, "CD":400,
    "C":100, "XC":90, "L":50, "XL":40,
    "X":10, "IX":9, "V":5, "IV":4, "I":1,
}

def roman_to_int_strict(s: str) -> Optional[int]:
    if not s: return None
    t = s.strip().upper()
    if not _ROMAN_CANONICAL_RE.match(t):
        return None
    i = 0; val = 0
    while i < len(t):
        # two-char token first
        if i+1 < len(t) and t[i:i+2] in _ROMAN_VAL:
            val += _ROMAN_VAL[t[i:i+2]]; i += 2
        else:
            val += _ROMAN_VAL[t[i]]; i += 1
    return val

# ラベル先頭抽出：= も許可、EOLも許可。Roman優先 → 数字 → レター
_LEADING_LABEL_RE = re.compile(
    r"""^\s*
        (?:final\s*answer|answer|ans\.?|option|choice)?\s*
        [:=,\-]?\s*
        [\(\[\{]?\s*
        (?P<label>[IVXLCDM]{1,6}|\d{1,2}|[A-Za-z])
        [\)\]\}]?
        \s*(?:[\.\:\-\)]\s*|\s+|$)     # ← EOLもOK
    """, flags=re.IGNORECASE | re.VERBOSE
)

_ANYWHERE_PAREN_LABEL_RE = re.compile(
    r"""[\(\[\{]\s*(?P<label>[IVXLCDM]{1,6}|\d{1,2}|[A-Za-z])\s*[\)\]\}]""",
    flags=re.IGNORECASE | re.VERBOSE
)

def compute_score(solution_str, ground_truth, **kwargs) -> float:
    """
    MCQ報酬:
      - ラベル同一性を最優先（A/B/C..., I/II/III..., 1/2/3... 相互対応）
      - 片側のみラベル抽出できた場合に限り、本文テキスト一致で救済（設定で制御）
      - 正解時は長さ減衰: exp(-alpha * z)。不正解は wrong_penalty
      - 形式ボーナス: <think>..., 純ラベル '#### B' に +bonus
    主要kwargs（既定値は従来どおり＋新規1つ）:
      text_fallback_requires_gold_body: bool = True
    """
    # ─ args ─
    alpha                 = kwargs.get("alpha", 0.05)
    wrong_penalty         = kwargs.get("wrong_penalty", -1.0)
    format_bonus          = kwargs.get("format_bonus", 0.1)
    hash_bonus            = kwargs.get("hash_bonus", 0.05)
    apply_bonus_incorrect = kwargs.get("apply_bonus_to_incorrect", True)
    clamp_positive_z      = kwargs.get("clamp_positive_z", True)
    clip_lo, clip_hi      = kwargs.get("clip", (-1.0, 1.0))

    length_mean           = kwargs.get("length_mean", None)
    length_std            = kwargs.get("length_std", None)
    correct_lengths       = kwargs.get("correct_lengths", None)
    include_self          = kwargs.get("include_self_if_correct", True)

    prefer_last_hash_line = kwargs.get("prefer_last_hash_line", True)
    allow_text_fallback   = kwargs.get("allow_text_fallback", True)
    strip_think_for_eval  = kwargs.get("strip_think_for_eval", True)
    letters_upper_bound   = int(kwargs.get("letters_upper_bound", 26))
    text_fallback_requires_gold_body = kwargs.get("text_fallback_requires_gold_body", True)

    # ─ helpers ─
    def has_think(s: str) -> bool:
        return bool(re.search(r"<think>.*?</think>", s or "", flags=re.DOTALL))

    def strip_think(s: str) -> str:
        return re.sub(r"<think>.*?</think>", "", s or "", flags=re.DOTALL)

    def strip_md(s: str) -> str:
        return re.sub(r"(\*\*|__|~~|`)", "", s or "").strip()

    def norm_spaces(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def get_final_target_text(s: str) -> str:
        t = s or ""
        if strip_think_for_eval:
            t = strip_think(t)
        t = t.strip()
        candidates = _HASH_RE.findall(t)
        if candidates:
            return candidates[-1].strip() if prefer_last_hash_line else candidates[0].strip()
        return t

    def text_key(s: str) -> str:
        if s is None: return ""
        s = strip_md(s)
        s = re.sub(r"^(final\s*answer|answer|ans\.?|option|choice)\s*[:=\-]\s*", "", s, flags=re.I)
        s = re.sub(r"^[\(\[\{]\s*([A-Za-z0-9IVXLCDM]+)\s*[\)\]\}]\s*$", r"\1", s)
        s = re.sub(r"[^\w\s\-\.]", " ", s)
        return norm_spaces(s.lower())

    def label_to_index(token: str) -> Optional[int]:
        if not token: return None
        t = token.strip().upper()

        # 1) Roman（優先）
        rv = roman_to_int_strict(t)
        if rv is not None and 1 <= rv <= letters_upper_bound:
            return rv - 1

        # 2) Numbers
        if re.fullmatch(r"\d{1,2}", t):
            val = int(t)
            if 1 <= val <= letters_upper_bound:
                return val - 1

        # 3) Letters
        if re.fullmatch(r"[A-Z]", t):
            idx = ord(t) - ord("A")
            return idx if 0 <= idx < letters_upper_bound else None

        return None

    def extract_label_and_text(s: str) -> Tuple[Optional[int], str, str]:
        if not s:
            return None, "", ""
        raw = strip_md(s)

        m = _LEADING_LABEL_RE.match(raw)
        if m:
            lbl = m.group("label")
            idx = label_to_index(lbl)
            rest = raw[m.end():].strip()
            return idx, rest, text_key(rest)

        m = _ANYWHERE_PAREN_LABEL_RE.search(raw)
        if m:
            lbl = m.group("label")
            idx = label_to_index(lbl)
            rest = (raw[:m.start()] + raw[m.end():]).strip()
            return idx, rest, text_key(rest)

        return None, raw, text_key(raw)

    def is_pure_label(s: str) -> bool:
        """ラベル＋些末記号のみ（本文なし）"""
        idx, rest, key = extract_label_and_text(s)
        return (idx is not None) and (key == "")

    # ─ 判定 ─
    def is_correct_mcq(pred_text: str, gold_text: str) -> bool:
        pred_core = get_final_target_text(pred_text)
        gold_core = get_final_target_text(gold_text)

        p_idx, p_rest, p_key = extract_label_and_text(pred_core)
        g_idx, g_rest, g_key = extract_label_and_text(gold_core)

        # 1) ラベルで一致（最優先）
        if (p_idx is not None) and (g_idx is not None):
            return p_idx == g_idx

        # 2) 本文フォールバック（gold に本文がある場合に限定）
        if allow_text_fallback:
            gold_has_body = (g_idx is not None and g_rest.strip() != "")
            if (not text_fallback_requires_gold_body) or gold_has_body:
                if p_key and g_key and (p_key == g_key):
                    return True

        # 3) 双方ラベル不明なら完全一致
        if (p_idx is None) and (g_idx is None):
            return (p_key != "") and (p_key == g_key)

        return False

    # ─ 実行 ─
    text_raw = solution_str if isinstance(solution_str, str) else str(solution_str)
    gt_raw   = ground_truth if isinstance(ground_truth, str) else str(ground_truth)

    correct = is_correct_mcq(text_raw, gt_raw)

    # ─ 長さ z ─
    L = len(text_raw or "")
    mu = length_mean; sd = length_std
    if (mu is None or sd is None) and (correct_lengths is not None):
        ref = list(correct_lengths)
        if correct and include_self:
            ref.append(L)
        if len(ref) == 0:
            mu, sd = None, None
        elif len(ref) == 1:
            mu, sd = float(ref[0]), 0.0
        else:
            mu = sum(ref) / len(ref)
            var = sum((v - mu) * (v - mu) for v in ref) / (len(ref) - 1)
            sd = var ** 0.5

    z = 0.0
    if correct and (mu is not None) and (sd is not None) and sd > 1e-8:
        z = (L - float(mu)) / float(sd)
        if clamp_positive_z and z < 0.0:
            z = 0.0

    # ─ 形式ボーナス ─
    reward = wrong_penalty if not correct else math.exp(-alpha * z)

    # gold 側の <think> は通常入らないので model 側だけ見れば十分
    if has_think(text_raw) and (apply_bonus_incorrect or correct):
        reward += float(format_bonus)
    
    hash_bonus = kwargs.get("hash_bonus", 0.1)
    if re.search(r"^\s*####\s*", text_raw or "", flags=re.MULTILINE):
        reward += hash_bonus

    # 純ラベル '#### B' 等を素直に加点
    pred_core = get_final_target_text(text_raw)
    if is_pure_label(pred_core) and (apply_bonus_incorrect or correct):
        reward += float(hash_bonus)

    # クリップ
    reward = max(clip_lo, min(clip_hi, reward))
    return float(reward)
