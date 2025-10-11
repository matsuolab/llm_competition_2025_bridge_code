# Model Management

このディレクトリは、MLモデルの設定ファイルと管理スクリプトを含みます。
実際のモデルファイル（.safetensors等）はGitリポジトリには含まれず、必要に応じて自動ダウンロードされます。

## ディレクトリ構造

```
train/ml_ops/models/
├── README.md                    # このファイル
├── model_manager.py            # モデル管理スクリプト
├── {model_name_A}/             # モデルAのディレクトリ
│   ├── model_config.yaml       # モデル設定ファイル（Git管理対象）
│   ├── config.json             # Hugging Face設定（Git管理対象）
│   ├── tokenizer_config.json   # トークナイザー設定（Git管理対象）
│   ├── vocab.json              # 語彙ファイル（Git管理対象）
│   ├── merges.txt              # マージファイル（Git管理対象）
│   ├── model-*.safetensors     # 実際のモデルファイル（Git管理対象外）
│   └── ...
├── {model_name_B}/             # モデルBのディレクトリ
│   ├── model_config.yaml
│   └── ...
└── ...
```

## Git管理ポリシー

### ✅ Git管理対象（リポジトリに含まれる）
- `model_config.yaml` - モデル設定とメタデータ
- `config.json` - Hugging Face設定ファイル
- `tokenizer_config.json` - トークナイザー設定
- `vocab.json`, `merges.txt` - 語彙・マージファイル
- `*.md` - ドキュメントファイル
- `model.safetensors.index.json` - モデルインデックス

### ❌ Git管理対象外（自動ダウンロード）
- `model-*.safetensors` - 実際のモデル重み
- `*.bin` - バイナリモデルファイル
- `*.pt`, `*.pth` - PyTorchモデルファイル
- その他の大きなファイル

## 使用方法

### 1. 既存モデルの使用

```python
from train.ml_ops.models.model_manager import ModelManager

# モデルマネージャーを初期化
manager = ModelManager()

# 利用可能なモデルを確認
print(manager.list_available_models())

# モデルをダウンロード・ロード
model, tokenizer = manager.load_model("Qwen3-4B-SFT-TEST2")

# モデルだけダウンロード（ロードしない）
manager.download_model("Qwen3-4B-SFT-TEST2")
```

### 2. 新しいモデルの追加

#### 方法A: 手動で設定ファイル作成

1. `ml_ops/models/your-model-name/` ディレクトリを作成
2. `model_config.yaml` を作成：

```yaml
model_name: "your-model-name"
huggingface_repo: "organization/model-name"
model_config:
  torch_dtype: "bfloat16"
  device_map: "auto"
  trust_remote_code: true
```

3. 設定ファイルをコミット
4. 実際のモデルファイルは自動ダウンロードされる

#### 方法B: プログラムで作成

```python
manager = ModelManager()
config_file = manager.create_model_config(
    model_name="your-model-name",
    huggingface_repo="organization/model-name",
    model_type="llama",
    model_config={
        "torch_dtype": "bfloat16",
        "device_map": "auto"
    }
)
```

### 3. モデル情報の確認

```python
# モデル情報を取得
info = manager.get_model_info("Qwen3-4B-SFT-TEST2")
print(f"Model: {info['name']}")
print(f"HF Repo: {info['huggingface_repo']}")
print(f"Local Path: {info['local_path']}")
```

## 利点

- ✅ **軽量リポジトリ**: 大きなモデルファイルはリポジトリに含まれない
- ✅ **自動管理**: 必要に応じてモデルが自動ダウンロードされる
- ✅ **設定管理**: モデルの設定とメタデータはバージョン管理される
- ✅ **再現性**: 同じ設定で誰でもモデルを使用可能
- ✅ **柔軟性**: Hugging Face Hub以外のソースにも対応可能
- ✅ **整理された構造**: モデルごとに独立したディレクトリ

## 注意事項

- 初回使用時はインターネット接続が必要
- モデルファイルは各モデルディレクトリにダウンロードされる
- 大きなモデルの場合、ダウンロードに時間がかかる場合がある
- 設定ファイル（.yaml）の変更後は、`force_download=True`で再ダウンロード可能
