import os
import pandas as pd
from tqdm import tqdm
import datasets
from huggingface_hub import upload_file


# --- Hugging Face Upload Settings: Repository Name ---
OUTPUT_DATASET_ID = "neko-llm/HLE_SFT_PHYBench"

def main():
    """
    HLE_SFT_PHYBench Dataset のデータクリーニングを行う Main Script。

    NOTE:
    関数の機能:
    1. HLE_SFT_PHYBench Dataset を読み込む。
    2. HLE_SFT_PHYBench Dataset の output の Prefix の <think> と Suffix の </think> を削除する。
    3. HLE_SFT_PHYBench Dataset の output の Prefix に <think></think> を追加する。
    4. 作成したデータセットを Hugging Face にアップロードする。
    """
    print("🚀 HLE_SFT_PHYBench Dataset のデータクリーニングを開始します...")

    # 出力ファイル設定
    output_filename = "HLE_SFT_PHYBench.csv"

    # --- Resume Logic ---
    processed_ids = []
    if os.path.exists(output_filename):
        print(f"📄 既存の結果ファイルを発見: '{output_filename}'")
        try:
            existing_df = pd.read_csv(output_filename)
            if 'id' in existing_df.columns:
                processed_ids = existing_df['id'].dropna().tolist()
                print(f"✅ {len(processed_ids)} 件の処理済み問題を発見。これらはスキップされます。")
        except pd.errors.EmptyDataError:
            print("⚠️ 既存の結果ファイルが空です。新規で開始します。")
        except Exception as e:
            print(f"⚠️ 既存ファイルの読み込みに失敗。新規で開始します。エラー: {e}")

    # --- HLE_SFT_PHYBench Dataset Loading ---
    print("📥 HLE_SFT_PHYBench Dataset を読み込み中...")
    try:
        # HLE_SFT_PHYBench Dataset を読み込み（trainデータのみ）
        olympiadbench_dataset = datasets.load_dataset("neko-llm/HLE_SFT_PHYBench", split='train')
        print(f"✅ データセット読み込み完了。{len(olympiadbench_dataset)} 件の問題を発見。")

        # サンプルデータの確認
        print("📋 サンプルデータ:")
        sample = olympiadbench_dataset[0]
        for key in sample.keys():
            print(f"  {key}: {str(sample[key])[:100]}...")

    except Exception as e:
        print(f"❌ データセット読み込みエラー: {e}")
        return


    # --- データクリーニング ---
    print("🧹 データクリーニングを開始します...")
    cleaned_data = []

    for i in tqdm(range(len(olympiadbench_dataset))):
        sample = olympiadbench_dataset[i]

        # NOTE: output の Prefix の <think> と Suffix の </think> を削除
        output_text = sample['output'].replace('<think>', '').replace('</think>', '')

        # NOTE: output の Prefix に <think></think> を追加
        output_text = f"<think></think>{output_text}"

        # クリーニングされたデータを作成
        cleaned_sample = {
            'id': sample['id'],
            'question': sample['question'],
            'output': output_text,
            'answer': sample['answer']
        }
        cleaned_data.append(cleaned_sample)

    # --- クリーニングされたデータを保存 ---
    print(f"💾 クリーニングされたデータを保存中: {output_filename}")
    cleaned_df = pd.DataFrame(cleaned_data)
    cleaned_df.to_csv(output_filename, index=False, encoding='utf-8')

    print(f"✅ データクリーニング完了!")
    print(f"   - 処理されたレコード数: {len(cleaned_data)}")
    print(f"   - 保存先: {output_filename}")

    # サンプルデータの確認
    print("\n📋 クリーニング後のサンプルデータ:")
    for i in range(min(3, len(cleaned_data))):
        sample = cleaned_data[i]
        print(f"  サンプル {i+1}:")
        print(f"    ID: {sample['id']}")
        print(f"    Answer: {repr(sample['answer'])}")
        print(f"    Output (先頭100文字): {repr(sample['output'][:100])}...")

    # --- 複数フォーマットでの保存とHugging Face アップロード ---
    try:
        print("\n🔄 最終データの処理とアップロードを開始...")
        final_df = cleaned_df  # 既にDataFrameとして準備されている

        # 統計情報の計算
        success_count = len(final_df)
        total_count = len(cleaned_data)

        print(f"📊 最終データセット統計:")
        print(f"   保存されたレコード数: {success_count}")
        print(f"   処理対象総数: {total_count}")
        print(f"   データセット品質: 100% (クリーニング済みデータ)")

        # ベースファイル名（拡張子なし）を取得
        base_filename = os.path.splitext(output_filename)[0]

        # --- 2. Parquet形式での保存 ---
        parquet_filename = f"{base_filename}.parquet"
        print(f"💾 Parquet形式で保存中: {parquet_filename}")
        final_df.to_parquet(parquet_filename, index=False)

        # --- 3. JSONL形式での保存 ---
        jsonl_filename = f"{base_filename}.jsonl"
        print(f"💾 JSONL形式で保存中: {jsonl_filename}")
        final_df.to_json(jsonl_filename, orient='records', lines=True, force_ascii=False)

        # --- Hugging Face Datasetsに3つのファイル形式でアップロード ---
        print(f"📤 {OUTPUT_DATASET_ID} に3つのファイル形式でアップロード中...")

        # 1. CSV ファイルをアップロード
        print("  📄 CSV ファイルをアップロード中...")
        upload_file(
            path_or_fileobj=output_filename,
            path_in_repo=output_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        # 2. Parquet ファイルをアップロード
        print("  📊 Parquet ファイルをアップロード中...")
        upload_file(
            path_or_fileobj=parquet_filename,
            path_in_repo=parquet_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        # 3. JSONL ファイルをアップロード
        print("  📝 JSONL ファイルをアップロード中...")
        upload_file(
            path_or_fileobj=jsonl_filename,
            path_in_repo=jsonl_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        print(f"✅ 3つのファイル形式でのアップロード完了: {OUTPUT_DATASET_ID}")

        # --- 統計情報の表示 ---
        print(f"\n📊 処理統計:")
        print(f"  - 総処理数: {total_count}")
        print(f"  - 成功数: {success_count}")
        print(f"  - 成功率: {success_count/total_count*100:.1f}%")

        print(f"\n📁 保存されたファイル:")
        print(f"  - CSV: {output_filename}")
        print(f"  - Parquet: {parquet_filename}")
        print(f"  - JSONL: {jsonl_filename}")

        print(f"\n🌐 Hugging Face Datasetsアップロード先:")
        print(f"  - Repository: {OUTPUT_DATASET_ID}")
        print(f"    ├─ {output_filename}")
        print(f"    ├─ {parquet_filename}")
        print(f"    └─ {jsonl_filename}")

    except Exception as e:
        print(f"❌ データ処理またはHugging Face Hubへのアップロードに失敗: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Processing complete ---")

if __name__ == "__main__":
    main()
