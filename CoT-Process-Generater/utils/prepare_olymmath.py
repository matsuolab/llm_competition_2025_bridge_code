import os
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm

def filter_and_upload_en_subsets(
    source_dataset_id: str,
    target_dataset_id: str,
    hf_token: str = None
):
    """
    特定のデータセットから_enで終わるsubsetをフィルタリングして
    新しいリポジトリに単一のtestスプリットとしてアップロードする
    
    Args:
        source_dataset_id: ソースデータセットのID (例: "username/dataset-name")
        target_dataset_id: ターゲットデータセットのID (例: "username/new-dataset-name")
        hf_token: HuggingFace APIトークン（オプション）
    """
    
    # HuggingFace APIの初期化
    api = HfApi(token=hf_token)
    
    # ターゲットリポジトリの作成（既に存在する場合はスキップ）
    try:
        create_repo(
            repo_id=target_dataset_id,
            repo_type="dataset",
            token=hf_token,
            exist_ok=True
        )
        print(f"✅ Repository created/confirmed: {target_dataset_id}")
    except Exception as e:
        print(f"⚠️ Repository creation note: {e}")
    
    # 利用可能なsubsetのリストを取得
    from datasets import get_dataset_config_names
    
    try:
        all_subsets = get_dataset_config_names(source_dataset_id)
        print(f"📋 Found {len(all_subsets)} subsets total")
        
        # _enで終わるsubsetをフィルタリング
        en_subsets = [subset for subset in all_subsets if subset.startswith("zh-hard")]
        print(f"🔍 Found {len(en_subsets)} English subsets:")
        for subset in en_subsets:
            print(f"   - {subset}")
        
        if not en_subsets:
            print("❌ No English subsets found!")
            return
        
        # 各English subsetを処理して一つのリストに集める
        all_datasets = []
        
        for subset in tqdm(en_subsets, desc="Loading subsets"):
            try:
                # testスプリットのみを読み込む
                dataset = load_dataset(
                    source_dataset_id,
                    subset,
                    split="test",
                    trust_remote_code=True  # カスタムスクリプトの場合に必要
                )
                
                # 元のsubset名を保持するために新しいカラムを追加（オプション）
                dataset = dataset.add_column("original_subset", [subset] * len(dataset))
                
                all_datasets.append(dataset)
                print(f"✅ Loaded {subset}: {len(dataset)} rows")
                
            except Exception as e:
                print(f"❌ Failed to load {subset}: {e}")
                continue
        
        if not all_datasets:
            print("❌ No data could be loaded!")
            return
        
        # 全てのデータセットを結合
        print(f"\n🔄 Concatenating {len(all_datasets)} datasets...")
        combined_dataset = concatenate_datasets(all_datasets)
        print(f"📊 Total rows in combined dataset: {len(combined_dataset)}")
        
        # データを新しいリポジトリにアップロード（subsetなし、testスプリットのみ）
        print(f"\n📤 Uploading to {target_dataset_id}...")
        
        try:
            combined_dataset.push_to_hub(
                target_dataset_id,
                split="test",  # testスプリットとしてアップロード
                token=hf_token
            )
            print(f"✅ Successfully uploaded combined dataset as 'test' split")
            
        except Exception as e:
            print(f"❌ Failed to upload combined dataset: {e}")
            return
        
        print(f"\n🎉 Successfully processed and uploaded {len(en_subsets)} English subsets")
        print(f"   Total rows: {len(combined_dataset)}")
        print(f"   Repository: {target_dataset_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTrying alternative approach with manual subset list...")
        
        # 代替アプローチ：手動でsubsetリストを指定
        manual_en_subsets = [
            "en-hard",
        ]
        
        process_manual_subsets_combined(
            source_dataset_id,
            target_dataset_id,
            manual_en_subsets,
            hf_token
        )

def process_manual_subsets_combined(source_id, target_id, subset_list, token):
    """手動で指定されたsubsetリストを処理して結合"""
    
    all_datasets = []
    
    for subset in tqdm(subset_list, desc="Loading manual subset list"):
        try:
            dataset = load_dataset(
                source_id,
                subset,
                split="test",
                trust_remote_code=True
            )
            
            # 元のsubset名を保持
            dataset = dataset.add_column("original_subset", [subset] * len(dataset))
            
            all_datasets.append(dataset)
            print(f"✅ Loaded {subset}: {len(dataset)} rows")
        except Exception as e:
            print(f"⚠️ Skipping {subset}: {e}")
    
    if all_datasets:
        # データセットを結合
        print(f"\n🔄 Concatenating {len(all_datasets)} datasets...")
        combined_dataset = concatenate_datasets(all_datasets)
        print(f"📊 Total rows: {len(combined_dataset)}")
        
        try:
            combined_dataset.push_to_hub(
                target_id,
                split="test",
                token=token
            )
            print(f"✅ Successfully uploaded combined dataset as 'test' split")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
    else:
        print("❌ No datasets could be loaded!")

# 使用例
if __name__ == "__main__":
    
    # ソースとターゲットのデータセットIDを指定
    SOURCE_DATASET = "RUC-AIBOX/OlymMATH"  # 実際のデータセットIDに置き換え
    TARGET_DATASET = "llm-2025-sahara/OlymMATH-en"  # 新しいデータセットID
    
    # 実行
    filter_and_upload_en_subsets(
        source_dataset_id=SOURCE_DATASET,
        target_dataset_id=TARGET_DATASET,
        hf_token=""
    )