import torch
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPUが利用できないためスキップ")
class TestCustomBuildLibraries:
    def test_import_apex(self):
        """ビルドしたapexがインポートできるかテスト"""
        try:
            import apex
            print(f"✓ apex version: {getattr(apex, '__version__', 'N/A')}")
        except ImportError as e:
            pytest.fail(f"apexのインポートに失敗しました: {e}")

    def test_import_transformer_engine(self):
        """ビルドしたtransformer_engineがインポートできるかテスト"""
        try:
            import transformer_engine
            print(f"✓ transformer_engine version: {transformer_engine.__version__}")
        except ImportError as e:
            pytest.fail(f"transformer_engineのインポートに失敗しました: {e}")

    def test_import_flash_attention(self):
        """ビルドしたflash_attnがインポートできるかテスト"""
        try:
            import flash_attn
            print(f"✓ flash_attn version: {flash_attn.__version__}")
        except ImportError as e:
            pytest.fail(f"flash_attnのインポートに失敗しました: {e}")

    def test_flash_attention_functionality(self):
        """FlashAttentionの基本関数が動作するかテスト"""
        try:
            from flash_attn.flash_attn_interface import flash_attn_func

            # (バッチサイズ, ヘッド数, シーケンス長, ヘッド次元)
            q = torch.randn(2, 4, 16, 64, dtype=torch.float16, device='cuda', requires_grad=True)
            k = torch.randn(2, 4, 16, 64, dtype=torch.float16, device='cuda', requires_grad=True)
            v = torch.randn(2, 4, 16, 64, dtype=torch.float16, device='cuda', requires_grad=True)
            
            output = flash_attn_func(q, k, v, causal=True)
            
            assert output.shape == q.shape, "出力テンソルの形状が不正です"
            assert output.dtype == torch.float16, "出力テンソルのデータ型が不正です"
            assert not torch.isnan(output).any(), "FlashAttentionの出力にNaNが含まれています"

            # 簡単な後方伝播もテスト
            output.sum().backward()
            assert q.grad is not None, "qの勾配が計算されていません"

        except Exception as e:
            pytest.fail(f"flash_attn_funcの実行中にエラーが発生しました: {e}")