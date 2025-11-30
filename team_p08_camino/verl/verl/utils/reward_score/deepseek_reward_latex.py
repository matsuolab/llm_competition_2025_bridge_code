###
#phi4-reasoningの報酬関数を実装
###
import math
import re
import signal
from collections import Counter

from sympy.parsing.latex import parse_latex
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("microsoft/Phi-4-reasoning-plus")

# 論文のセクション4.1および4.2から引用した定数
# 報酬関数で使用する長さのパラメータ
L_MAX = 31744
L_POS_CONTROL = 25600
L_NEG_CONTROL = 3702

# 報酬値の範囲
R_MAX_POS = 1.0
R_MIN_POS = 0.5
R_MAX_NEG = -0.5
R_MIN_NEG = -1.0

# 最終的な報酬の重み
W_ACC = 8 / 13
W_REP = 1 / 13

# 繰り返しペナルティのパラメータ
NGRAM_SIZE = 5
NGRAM_FREQ_THRESHOLD = 5
_SOLUTION_CLIP_CHARS = 300

# タイムアウト時に呼び出され、例外を発生させる関数
def timeout_handler(signum, frame):
    raise TimeoutError("処理がタイムアウトしました。")
def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("####(\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def find_last_boxed_content(text: str) -> str:
    """
    文字列中の最後の "\\boxed{...}" の中身を、入れ子括弧を考慮して抽出します。
    エスケープされた括弧 \{ や \} は無視します。
    """
    try:
        # 最後の "\\boxed{" の開始インデックスを探します
        last_boxed_start_index = text.rfind("\\boxed{")
        if last_boxed_start_index == -1:
            return ""

        # コンテンツの実際の開始位置
        content_start_index = last_boxed_start_index + len("\\boxed{")

        # 対応する閉じ括弧 '}' を探します
        brace_level = 1
        for i in range(content_start_index, len(text)):
            char = text[i]

            # LaTeXでエスケープされた括弧 \{ や \} はレベル計算に含めません
            if text[i-1] == '\\' and (char == '{' or char == '}'):
                continue
            
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
            
            # brace_levelが0になったら、それが対応する閉じ括弧です
            if brace_level == 0:
                return text[content_start_index:i]
        
        # 最後まで見ても対応する閉じ括弧が見つからなかった場合
        return ""

    except Exception:
        # 何らかのエラーが発生した場合
        return ""
def extract_thought_and_answer(solution_str: str) -> tuple[str, str, bool]:
    """
    文字列から<think>...</think>と最後の\\boxed{...}を抽出します。
    \\boxed{...}内の入れ子括弧に対応しています。
    """
    # <think>...</think> の抽出ロジックは変更ありません
    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)

    if think_match:
        thinking_process = think_match.group(1).strip()
        is_format_valid = True
    else:
        thinking_process = ""
        is_format_valid = False

    # \\boxed{...} の抽出を新しい堅牢な関数に置き換えます
    answer = find_last_boxed_content(solution_str)

    return thinking_process, answer, is_format_valid
def parse_solution(solution_str: str) -> tuple[str | None, str | None, bool]:
    """
    <think>...</think>{answer} という構造の文字列を解析する。

    Args:
        solution_str: モデルの生成出力。

    Returns:
        タプル (thinking_process, answer, is_format_valid)
    """
    # 正規表現を使用して<think>ブロックとそれに続く回答を抽出
    match = re.search(r"<think>(.*?)</think>(.*)", solution_str, re.DOTALL)

    if match:
        thinking_process = match.group(1).strip()
        answer = match.group(2).strip()
        return thinking_process, answer, True
    else:
        # thinkタグが見つからない、または形式が不正
        return None, None, False

def _compute_repetition_penalty(text: str) -> float:
    """
    n-gramの頻度に基づいて繰り返しペナルティを計算します。
    """
    words = text.split()
    if len(words) < NGRAM_SIZE:
        return 0.0
    # n-gramを生成
    ngrams = [" ".join(words[i:i+NGRAM_SIZE]) for i in range(len(words) - NGRAM_SIZE + 1)]
    if not ngrams:
        return 0.0

    ngram_counts = Counter(ngrams)
    frequent_ngrams = {k: v for k, v in ngram_counts.items() if v > NGRAM_FREQ_THRESHOLD}

    if not frequent_ngrams:
        return 0.0

    term1 = len(frequent_ngrams) / len(ngrams)
    max_freq = max(frequent_ngrams.values())
    total_possible_ngrams = len(words) / NGRAM_SIZE if len(words) > 0 else 1
    term2 = max_freq / total_possible_ngrams

    penalty = -max(term1, term2)
    return penalty

def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    """
    Phi-4-reasoning論文で説明されている報酬関数に基づいて最終的なスコアを計算します。

    Args:
        solution_str: モデルから生成された完全なテキスト。(tokenizedではなく、文字列形式)
        ground_truth: 正解。
        data_source: データソースの名前。現在は "gsm8k" のみ対応。
        
    """
    # 1. 出力文字列を解析し、フォーマットを検証
    thinking_process, answer, is_format_valid = extract_thought_and_answer(solution_str)
    L=len(TOKENIZER.tokenize(solution_str))
    print("---solution_str---")
    print(solution_str)  # Debugging output

    signal.signal(signal.SIGALRM, timeout_handler)
    # 30秒でタイムアウトするように設定(必要に応じて変更)
    signal.alarm(30)
    try: 
        latex_answer=parse_latex(str(answer).lower(),backend="lark")
        latex_ground_truth=parse_latex(str(ground_truth).lower(),backend="lark")
    except Exception as e:  # 必要に応じて全ての例外をキャッチ
        latex_answer=str(answer).lower().replace(" ","")
        latex_ground_truth=str(ground_truth).lower().replace(" ","")
    finally:
        signal.alarm(0)
    print("---is_format_valid---")
    print(is_format_valid)
    print("---answer---")
    print(answer)  # Debugging output
    print("---ground_truth---")
    print(ground_truth)  # Debugging output    
    # 2. フォーマット違反のオーバーライドを処理
    # <think>タグが不正な場合は is_format_valid が False になる
    # answerが不適切な形の場合もフォーマット違反とする 
    if not is_format_valid:
        r_acc_scaled = -1.0
    # 生成が不完全な場合
    elif L >= L_MAX-1:
        # imcomplete(eostokenなし)はこの関数では厳密な実装はできないので，max_lengthを超えた場合にフォーマット違反として扱う
        # ここでは、L_MAXを超える場合にフォーマット違反として扱う
		# (Lは開始トークンおよび終了トークンを含まず，L_MAXは終了トークンを含むため，L_MAX-1と比較)
        # TODO:imcompleteの完全な実装
        r_acc_scaled = -0.5
    else:
        # 3. 回答が正解かどうかを報酬に反映
        #ground_truthがlatex構文に適していなかった場合，元のanswerと比較する           
        is_correct= (latex_answer is not None and latex_answer == latex_ground_truth)
        # 注記: 論文ではトークン長が使用されていますが、ここでは単語数を代理として使用します。
        # 正確な実装には、トークナイザが必要です。
        #L = len(solution_str.split())
        L= len(TOKENIZER.tokenize(solution_str))

        if is_correct:
            rho_plus = min(1.0, max(0, L - L_POS_CONTROL) / (L_MAX - L_POS_CONTROL))
            cos_term = 0.5 * (R_MAX_POS - R_MIN_POS) * (1 + math.cos(math.pi * rho_plus))
            r_acc_scaled = R_MIN_POS + cos_term
        else:
            rho_minus = min(1.0, L / L_NEG_CONTROL)
            cos_term = 0.5 * (R_MIN_NEG - R_MAX_NEG) * (1 + math.cos(math.pi * rho_minus))
            r_acc_scaled = R_MAX_NEG + cos_term

    # 4. 繰り返しペナルティを計算 (文字列全体を対象)
    r_rep = _compute_repetition_penalty(solution_str)

    # 5. 最終的な重み付きスコアを計算
    final_score = (W_ACC * r_acc_scaled) + (W_REP * r_rep)
    print("---final_score---")
    print(final_score)  # Debugging output

    return final_score