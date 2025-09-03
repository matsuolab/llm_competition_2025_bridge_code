import re, math, json, os
from typing import Optional, Tuple

_VERIFY_FUNC = None
_MV_AVAILABLE = None
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    _MV_AVAILABLE = True
except Exception:
    _MV_AVAILABLE = False
    class TimeoutException(Exception):
        pass
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_HASH_LINE_RE = re.compile(r"^\s*####\s*(.+?)\s*$", flags=re.MULTILINE)

_BAD_FINAL_TOKENS = re.compile(
    r"(?i)\b(wait|hence|therefore|alternatively|let'?s|maybe|i think|proof|step|consider)\b"
)

# 入れ子の中括弧に対応した正規表現
_BOXED_RE = re.compile(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', flags=re.DOTALL)
_ANSWER_LABEL_RE = re.compile(r"(?i)(?:final answer|answer is|answer:)\s*[:\-]?\s*(.+)")
_NUM_RE = re.compile(r"^\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[1-9]\d*)?\s*$")
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")

# display 数式 / LaTeX 分数
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
_FRAC_TEX_RE     = re.compile(r"\\d?frac\{[^{}]+\}\{[^{}]+\}")

# 「数式らしくない」英語ノイズを弾くための簡易フィルタ
_BAD_WORDS_RE = re.compile(r"(?i)\b(second|term|step|analy[sz]e|proof|therefore|hence)\b")

def _is_plausible_final(s: Optional[str]) -> bool:
    if not s: 
        return False
    # \dfrac を \frac に正規化してからチェック
    s_normalized = s.replace(r'\dfrac', r'\frac')
    # \frac を含む or 数値/有理数として解釈できる
    if _FRAC_TEX_RE.search(s_normalized):
        return True
    if _NUM_RE.match(_latex_to_plain_number(s_normalized) or ""):
        return True
    # 明らかな英語見出しは除外
    if _BAD_WORDS_RE.search(s):
        return False
    return False

def _find_first_boxed_balanced(text: str) -> Optional[str]:
    """text 内の最初の \boxed{...}（入れ子対応）を返す"""
    if not text:
        return None
    m = _BOXED_OPEN_RE.search(text)
    if not m:
        return None
    open_idx = m.end() - 1
    depth, i, n = 1, open_idx + 1, len(text)
    while i < n and depth > 0:
        c = text[i]
        if c == "{": depth += 1
        elif c == "}": depth -= 1
        i += 1
    if depth != 0:
        return None
    result = text[open_idx + 1:i - 1].strip()
    return result.replace(r'\dfrac', r'\frac')

def _latex_to_plain_number(s: str) -> str:
    if not s:
        return s
    s = re.sub(r'\\boxed\{(.+?)\}', r'\1', s)
    s = re.sub(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', s)
    s = re.sub(r'\\cdot', '*', s)
    return s.strip()

def _extract_after_final_answer(text: str) -> Optional[str]:
    """最後の 'Final Answer' 以降、直後のブロックだけから最終解を抜く。"""
    if not text:
        return None

    # Final Answerのパターン（大文字小文字、Markdown装飾を考慮）
    # **Final Answer** や ### Final Answer なども認識
    final_pattern = re.compile(
        r"(?im)(?:^|\n)\s*(?:\*{1,2}\s*)?(?:#{1,6}\s*)?(?:\*{1,2}\s*)?final\s*answer(?:\*{1,2})?\s*:?\s*",
        re.IGNORECASE | re.MULTILINE
    )
    
    # 最後のFinal Answerを探す
    matches = list(final_pattern.finditer(text))
    if not matches:
        return None
    
    last_match = matches[-1]
    # Final Answerの後のテキスト
    after_text = text[last_match.end():].strip()
    
    # 同一行に答えがある場合
    first_line = after_text.split('\n')[0] if after_text else ""
    if first_line and not first_line.startswith('$'):
        # 行内に\boxedがある場合
        boxed_match = _BOXED_RE.search(first_line)
        if boxed_match:
            result = boxed_match.group(1).replace(r'\dfrac', r'\frac')
            if _is_plausible_final(result):
                return result
    
    # display math $$ ... $$ を探す
    dm = _DISPLAY_MATH_RE.search(after_text[:2000])
    if dm:
        inner = dm.group(1).strip().replace(r'\dfrac', r'\frac')
        # \boxed{...} があればそれを優先
        boxed_match = _BOXED_RE.search(inner)
        if boxed_match:
            result = boxed_match.group(1).replace(r'\dfrac', r'\frac')
            if _is_plausible_final(result):
                return result
        # \boxedがなくても数式として妥当なら返す
        if _is_plausible_final(inner):
            return inner

    # 直後のテキストから\boxed{...}を探す
    boxed_match = _BOXED_RE.search(after_text[:2000])
    if boxed_match:
        result = boxed_match.group(1).replace(r'\dfrac', r'\frac')
        if _is_plausible_final(result):
            return result

    # \frac を探す
    frac_match = _FRAC_TEX_RE.search(after_text[:500])
    if frac_match:
        result = frac_match.group(0)
        if _is_plausible_final(result):
            return result

    # 単独数値行
    for line in after_text[:500].splitlines()[:5]:  # 最初の5行だけチェック
        line = line.strip()
        if _NUM_RE.match(line):
            return line

    return None

def _extract_last_boxed_balanced(text: str) -> Optional[str]:
    """最後の \\boxed{...} を見つける（入れ子対応）"""
    if not text:
        return None
    
    # 入れ子対応の正規表現で全ての\boxed{...}を探す
    matches = _BOXED_RE.findall(text)
    if matches:
        result = matches[-1].strip()
        return result.replace(r'\dfrac', r'\frac')
    
    # 上記で見つからない場合、手動でバランスを取る方法を試す
    last = None
    for m in _BOXED_OPEN_RE.finditer(text):
        last = m
    if last is None:
        return None
    
    open_idx = last.end() - 1
    depth = 1
    i = open_idx + 1
    n = len(text)
    
    while i < n and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    
    if depth != 0:
        return None
    
    result = text[open_idx + 1:i - 1].strip()
    return result.replace(r'\dfrac', r'\frac')

def _strip_latex_wrappers(s: Optional[str]) -> Optional[str]:
    if not s: return s
    s = s.strip()
    changed = True
    while changed:
        changed = False
        for pat in (r'^\$\$(.*)\$\$$', r'^\$(.*)\$$', r'^\\\[(.*)\\\]$', r'^\\\((.*)\\\)$'):
            m = re.match(pat, s, flags=re.DOTALL)
            if m:
                s = m.group(1).strip(); changed = True
    return s

_RHS_SAFE_LHS = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,8}$")
def _rhs_if_eq(s: str) -> str:
    if not s or "=" not in s: return s
    lhs, rhs = s.rsplit("=", 1)
    return rhs.strip() if _RHS_SAFE_LHS.match(lhs.strip()) else s

def _normalize_sets(s: str) -> str:
    t = re.sub(r"[(){}\[\]]", " ", s)
    parts = re.split(r"\s*(?:,|;|\||and|or|，|、)\s*", t, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1: return s
    parts = [re.sub(r"[.\s]+$", "", p) for p in parts]
    return r"\{" + ",".join(sorted(parts)) + r"\}"

def _extract_final(text: str) -> Tuple[Optional[str], str]:
    """
    最終解（短い断片）と抽出モードを返す。
    優先順: Final Answer 以降 → 最後の完全な\boxed{} → #### → Answerラベル → "answer is"
    """
    # 'Final Answer' の後ろを最優先で探索
    fa = _extract_after_final_answer(text or "")
    if fa is not None:
        return fa.strip(), "final_mark"

    # 入れ子対応で最後の \boxed{...} を探す
    boxed = _extract_last_boxed_balanced(text or "")
    if boxed is not None:
        return boxed.strip(), "boxed"

    # #### <ans> の最後（優先度を下げる）
    m_hash = _HASH_LINE_RE.findall(text or "")
    if m_hash:
        # #### の後が数式っぽいものだけを採用
        result = m_hash[-1].strip()
        # 英語の見出しっぽいものは除外
        if not _BAD_WORDS_RE.search(result):
            return result.replace(r'\dfrac', r'\frac'), "hash"

    # "Final Answer:" 等（同一行にあるケース）
    m_lab = _ANSWER_LABEL_RE.findall(text or "")
    if m_lab:
        result = m_lab[-1].strip()
        return result.replace(r'\dfrac', r'\frac'), "label"

    # "the answer is ..." を緩く
    m_ans = re.findall(r"(?i)\bthe answer is\s*[:\-]?\s*([^\n]+)", text or "")
    if m_ans:
        result = m_ans[-1].strip()
        return result.replace(r'\dfrac', r'\frac'), "phrase"

    return None, "none"

def _is_numeric_like(s: str) -> bool:
    return bool(_NUM_RE.match(s or ""))

def _safe_float(x: str) -> Optional[float]:
    try:
        if "/" in x:
            num, den = x.split("/", 1)
            return float(num) / float(den)
        return float(x)
    except Exception:
        return None

def _numeric_shaping(pred: str, gold: str, lam=5.0) -> float:
    """
    数値/有理数に限定した"近さ"の微小ボーナス [0, 0.2]
    """
    if not (_is_numeric_like(pred) and _is_numeric_like(gold)): return 0.0
    pv, gv = _safe_float(pred), _safe_float(gold)
    if pv is None or gv is None: return 0.0
    if pv == gv: return 0.2
    d = abs(pv - gv)
    d_rel = d/abs(gv) if gv != 0 else d
    return 0.2 * math.exp(-lam * d_rel)

def compute_score(solution_str, ground_truth, **kwargs):
    """
    報酬 = 
      1) まず math-verify。v==1.0 なら即 +1
      2) v!=1.0 のときだけ Groq で最終解を採点（<think> 除去テキストを渡す）
         - Groq の 0..100 点を [−1, +1] に線形マップして基礎報酬に採用
      3) 近似/数値ボーナス、フォーマット/長さペナルティを合成 → クリップ
    """
    # 既存パラメータ（互換）
    timeout_score    = float(kwargs.get("timeout_score", 0.0))
    require_think    = bool(kwargs.get("require_think", True))
    require_hash_last= bool(kwargs.get("require_hash_last", True))
    missing_think_penalty = float(kwargs.get("missing_think_penalty", 0.20))
    missing_hash_penalty  = float(kwargs.get("missing_hash_penalty", 0.20))
    format_bonus     = float(kwargs.get("format_bonus", 0.03))
    apply_bonus_incorrect = bool(kwargs.get("apply_bonus_to_incorrect", True))
    wrap_gold        = bool(kwargs.get("wrap_gold_with_boxed", True))
    allow_set_norm   = bool(kwargs.get("allow_set_normalization", True))
    remove_think_for_ver = bool(kwargs.get("remove_think_for_verify", True))
    clip_lo, clip_hi = kwargs.get("clip", (-1.0, 1.0))

    # 新規/変更パラメータ
    gate_mode        = kwargs.get("gate_mode", "soft")
    think_softcap    = int(kwargs.get("think_softcap_chars", 4000))
    think_softcap_strength = float(kwargs.get("think_softcap_strength", 0.0))
    bad_final_max    = int(kwargs.get("bad_final_max_chars", 120))
    bad_final_pen    = float(kwargs.get("bad_final_penalty", 0.30))
    debug            = bool(kwargs.get("debug", False))

    # Groq 関連（追加）
    groq_enable      = bool(kwargs.get("groq_enable", False))
    groq_model       = kwargs.get("groq_model", "qwen/qwen3-32b")
    groq_correct_threshold = float(kwargs.get("groq_correct_threshold", 0.60))  # 60 点以上を正解扱い
    groq_max_tokens  = int(kwargs.get("groq_max_tokens", 128))
    groq_default_score = float(kwargs.get("judge_default_score", 0.0))  # 0..1
    groq_api_key = (
        kwargs.get("groq_api_key")
        or os.getenv(kwargs.get("groq_api_key_envvar", "GROQ_API_KEY"))
        or os.getenv("GROQ_API_KEY")
    )
    # Colab の userdata も一応見る
    if not groq_api_key:
        try:
            from google.colab import userdata as _colab_user
            groq_api_key = _colab_user.get('Groq_API_KEY') or _colab_user.get('GROQ_API_KEY')
        except Exception:
            pass

    # ------ テキスト処理ユーティリティ（既存） ------
    text_raw = solution_str if isinstance(solution_str, str) else str(solution_str)
    gt_raw   = ground_truth if isinstance(ground_truth, str) else str(ground_truth)

    def has_think(s: str) -> bool:
        return bool(re.search(r"<think>.*?</think>", s or "", flags=re.DOTALL))

    def extract_think(s: str) -> str:
        m = re.search(r"<think>(.*?)</think>", s or "", flags=re.DOTALL)
        return m.group(1) if m else ""

    def strip_think(s: str) -> str:
        return re.sub(r"<think>.*?</think>", "", s or "", flags=re.DOTALL)

    def is_hash_last(s: str) -> bool:
        lines = [ln.rstrip() for ln in (s or "").splitlines()]
        while lines and (not lines[-1] or lines[-1].isspace()):
            lines.pop()
        return bool(lines) and lines[-1].lstrip().startswith("####")

    def _prepare_for_verify(text: str) -> str:
        t = strip_think(text) if remove_think_for_ver else (text or "")
        final, mode = _extract_final(t)
        if debug:
            print(f"[prepare] extracted: mode={mode}, final='{final[:50] if final else None}...'")
        if final is None:
            final = t  # フォールバック
        final = final.replace(r'\dfrac', r'\frac')
        t = _strip_latex_wrappers(final) or ""
        t = _rhs_if_eq(t)
        if allow_set_norm:
            t = _normalize_sets(t)
        return t.strip()

    def verify_score(pred_text: str, gold_text: str) -> float:
        global _VERIFY_FUNC, _MV_AVAILABLE
        if not _MV_AVAILABLE:
            return 0.0
        if _VERIFY_FUNC is None:
            _VERIFY_FUNC = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
            )
        gold = _prepare_for_verify(gold_text)
        pred = _prepare_for_verify(pred_text)
        if wrap_gold and "\\boxed" not in gold:
            gold = f"\\boxed{{{gold}}}"
        if wrap_gold and "\\boxed" not in pred:
            pred = f"\\boxed{{{pred}}}"
        if debug:
            print(f"[verify] gold='{gold[:120]}' pred='{pred[:120]}'")
        try:
            score, _ = _VERIFY_FUNC([gold], [pred])
        except TimeoutException:
            return float(timeout_score)
        except Exception:
            return 0.0
        return float(score)

    # Groq 採点（v!=1.0 の時だけ呼ぶ・<think> 除去で投げる）
    def _build_groq_prompt(pred_final: str, gold_final: str) -> str:
    # ユーザーメッセージ用（思考や説明を一切許さない）
        return f"""
    Judge ONLY the two FINAL ANSWERS below for mathematical equivalence.

    Output exactly ONE line and nothing else:
    ★Score: 100   if equivalent
    ★Score: 0     otherwise

    Do NOT include analysis, explanations, markdown, or any extra text.

    [Model Final Answer]
    {pred_final}

    [Gold Final Answer]
    {gold_final}
    """.strip()

    def _parse_groq_score(text: str) -> Optional[float]:
        if not text:
            return None
        m = re.search(r"[★⭐]?\s*Score\s*[:：]\s*(\d{1,3})", text, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"\b(\d{1,3})\b", text)  # 最後の保険
        if not m:
            return None
        val = max(0, min(100, int(m.group(1))))
        return val / 100.0

    def groq_semantic_score(response_no_think: str, gold_answer_clean: str) -> float:
        if not groq_enable:
            return groq_default_score
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        prompt = _build_groq_prompt(response_no_think, gold_answer_clean)
        resp = client.chat.completions.create(
            model=groq_model,
            temperature=0,
            #max_tokens=groq_max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        content = (resp.choices[0].message.content or "").strip()
        if debug:
            print(f"[groq] raw='{content[:200]}'")
        s = _parse_groq_score(content)
        if s is None:
            return groq_default_score
        return float(s)

    # ------ フォーマットゲート ------
    fmt_penalty = 0.0
    if require_think and not has_think(text_raw):
        fmt_penalty -= missing_think_penalty
    if require_hash_last and not is_hash_last(text_raw):
        fmt_penalty -= missing_hash_penalty

    text_no_think = strip_think(text_raw)
    final_raw, mode = _extract_final(text_no_think)
    if debug:
        print(f"[extract] mode={mode} final_raw='{final_raw[:100] if final_raw else None}'")
    final_clean = _prepare_for_verify(text_raw)

    bad_final = False
    if not final_raw:
        bad_final = True
    else:
        if len(final_raw) > bad_final_max and not ("{" in final_raw or "=" in final_raw):
            bad_final = True
        if _BAD_FINAL_TOKENS.search(final_raw):
            bad_final = True

    if gate_mode == "hard" and (fmt_penalty < 0 or bad_final):
        return float(max(clip_lo, -1.0))

    # ------ 1) math-verify ------
    v = verify_score(text_raw, gt_raw)
    is_mv_perfect = (abs(v - 1.0) < 1e-12)  # ほぼ 1.0 を完璧扱い

    # ------ 近似（数値）ボーナス ------
    near_bonus = 0.0
    if not is_mv_perfect:
        near_bonus = _numeric_shaping(final_clean, _prepare_for_verify(gt_raw))
    
    # ------ Groq（v!=1.0 の時だけ） ------
    # groq_s = None
    # if not is_mv_perfect:
    #     groq_s = groq_semantic_score(text_no_think.strip(), _prepare_for_verify(gt_raw))

    # ------ 基礎報酬の決定 ------
    if is_mv_perfect:
        is_correct = True
        reward = 1.0
    else:
        # s = groq_s if groq_s is not None else 0.0  # 0..1
        # is_correct = (s >= groq_correct_threshold)
        # # 0..1 → [-1, +1] にマップ（0.5 が 0）
        # reward = 2.0 * s - 1.0
        is_correct = False
        reward = -1.0

    # ------ フォーマット小ボーナス ------
    if not bad_final:
        has_hash = is_hash_last(text_raw)
        if (has_think(text_raw) and (apply_bonus_incorrect or is_correct)):
            reward += format_bonus
        if (has_hash and (apply_bonus_incorrect or is_correct)):
            reward += format_bonus

    # 近似/LLM（Groq は基礎報酬に反映済み）・数値ボーナス
    reward += near_bonus

    # フォーマット/長さペナルティ
    think_len = len(extract_think(text_raw)) or len(text_raw)
    if think_softcap > 0:
        over = max(0, think_len - think_softcap)
        x = over / 1000.0
        len_pen = think_softcap_strength * (1.0 - math.exp(-x))
    else:
        len_pen = 0.0
    if bad_final:
        fmt_penalty -= bad_final_pen
    reward += fmt_penalty
    reward -= len_pen

    # クリップ
    reward = max(clip_lo, min(clip_hi, reward))
    return float(reward)
