###
#deepseek r1の報酬関数を実装
###
import math
import os
import re
import signal
import time
from collections import Counter

from groq import Groq
from sympy.parsing.latex import parse_latex


#from Levenshtein import ratio as levenshtein_ratio
# タイムアウト時に呼び出され、例外を発生させる関数
def timeout_handler(signum, frame):
    raise TimeoutError("処理がタイムアウトしました。")

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
def groq_complex_match(answer,ground_truth):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    system_input='''
    Are the `answer` and `ground_truth` below semantically the same? Answer only with `True` or `False` and Put your final answer inside \\boxed{}\n
    '''
    # Set the system prompt
    system_prompt = {
        "role": "system",
        "content": system_input
    }

    # Set the user prompt
    user_input = f'''###answer:\n
    {answer}\n
    ###ground_truth:\n
    {ground_truth}\n
    ###judge:\n
    '''
    user_prompt = {
        "role": "user", "content": user_input
    }

    # Initialize the chat history
    chat_history = [system_prompt, user_prompt]
    time.sleep(1.5)
    response = client.chat.completions.create(model="meta-llama/llama-4-scout-17b-16e-instruct",
                                                messages=chat_history,
                                                max_tokens=1000,
                                                temperature=0.01)

    # Print the response
    llm_judge=response.choices[0].message.content
    llm_judge=find_last_boxed_content(llm_judge)
    return llm_judge
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
    match = re.search(r"(.*?)</think>(.*)", solution_str, re.DOTALL)

    if match:
        thinking_process = match.group(1).strip()
        answer = match.group(2).strip()
        return thinking_process, answer, True
    else:
        # thinkタグが見つからない、または形式が不正
        return None, None, False


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    """
    Phi-4-reasoning論文で説明されている報酬関数に基づいて最終的なスコアを計算します。

    Args:
        solution_str: モデルから生成された完全なテキスト。(tokenizedではなく、文字列形式)
        ground_truth: 正解。
        is_incomplete: 生成がシーケンス終了トークンなしで不完全な場合はTrue。
    """
    # 1. 出力文字列を解析し、フォーマットを検証
    thinking_process, answer, is_format_valid = extract_thought_and_answer(solution_str)
    print("---solution_str---")
    print(solution_str)  # Debugging output
    print("---is_format_valid---")
    print(is_format_valid)
    print("---answer---")
    print(answer)  # Debugging output
    print("---ground_truth---")
    print(ground_truth)  # Debugging output
    #option: reasoningの正解を取得する
    #thinking_process_truth,_, is_format_valid_truth = parse_solution(truth_reasoning)
    #if is_format_valid_truth:
    #    truth_reasoning = thinking_process_truth
    # 2. フォーマット違反のオーバーライドを処理
    # <think>タグが不正な場合は is_format_valid が False になる
    if is_format_valid:
        r_format = 1.0
    else:
        r_format = 0
    # 3. 回答が正解かどうかを報酬に反映
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
    is_correct = (latex_answer is not None and latex_answer == latex_ground_truth)
    if not is_correct:
        signal.alarm(30)
        try:
            llm_correct_judge=groq_complex_match(answer,ground_truth)
        except Exception as e:
            llm_correct_judge="False"
        finally:
            signal.alarm(0)
        if llm_correct_judge=="True":
            is_correct=True
        else:
            is_correct=False
    if is_correct:
        # 正解の場合、正解度報酬を1.0にスケーリング
        r_acc_scaled = 1.0
    else:
        r_acc_scaled = 0.0  # Assign a default value for r_acc_scaled
    #(option) Levenshtein距離を使用してスコアを計算
    #r_leven = levenshtein_ratio(str(thinking_process), str(truth_reasoning))

    final_score = (r_format + r_acc_scaled)/2# 平均を取ることでスコアを正規化
    print("---final_score---")
    print(final_score)  # Debugging output
    return final_score