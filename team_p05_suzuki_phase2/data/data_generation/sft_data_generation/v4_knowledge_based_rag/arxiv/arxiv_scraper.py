#!/usr/bin/env python3
"""
ArXiv Scraper with License Checking for LLM Training
ライセンスチェック機能付きArXivスクレイパー
"""

#!/usr/bin/env python3
"""
ArXiv Scraper
"""

import time
import json
import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime, timedelta
import re
from enum import Enum
import gc
import pickle
import signal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LicenseType(Enum):
    """ライセンスタイプ"""
    CC_BY = "CC BY"  # ✅ 安全
    CC_BY_SA = "CC BY-SA"  # ✅ 安全（同一ライセンス継承）
    CC0 = "CC0"  # ✅ 安全（パブリックドメイン）
    CC_BY_NC = "CC BY-NC"  # ⚠️ 非商用のみ
    CC_BY_NC_SA = "CC BY-NC-SA"  # ⚠️ 非商用のみ
    CC_BY_NC_ND = "CC BY-NC-ND"  # ❌ 使用不可
    CC_BY_ND = "CC BY-ND"  # ❌ 使用不可
    ARXIV_DEFAULT = "ArXiv Non-exclusive"  # ⚠️ 要確認
    UNKNOWN = "Unknown"  # ❌ 使用不可


class ArxivScraper:
    """基本的なArXivスクレイパー"""
    
    # LLM訓練に安全なライセンス
    SAFE_LICENSES = [
        LicenseType.CC_BY,
        LicenseType.CC_BY_SA,
        LicenseType.CC0
    ]
    
    # 非商用研究でのみ使用可能
    RESEARCH_ONLY_LICENSES = [
        LicenseType.CC_BY_NC,
        LicenseType.CC_BY_NC_SA,
        LicenseType.ARXIV_DEFAULT
    ]
    
    # カテゴリ分布（簡易版）
    CATEGORY_DISTRIBUTION = {
        'mathematics': {
            'categories': ['math.AG', 'math.AT', 'math.AP', 'math.CT', 'math.CA', 
                          'math.CO', 'math.AC', 'math.CV', 'math.DG', 'math.DS'],
            'percentage': 0.41
        },
        'physics': {
            'categories': ['physics.acc-ph', 'physics.ao-ph', 'physics.atom-ph'],
            'percentage': 0.09
        },
        'computer_science': {
            'categories': ['cs.AI', 'cs.AR', 'cs.CC', 'cs.CE', 'cs.CG', 'cs.CL',
                          'cs.CR', 'cs.CV', 'cs.CY', 'cs.DB', 'cs.DC', 'cs.DL',
                          'cs.LG', 'cs.LO'],
            'percentage': 0.10
        },
        'other': {
            'categories': ['stat.AP', 'stat.CO', 'stat.ME', 'stat.ML'],
            'percentage': 0.09
        }
    }
    
    def __init__(self, commercial_use=False, max_papers=100, categories=None,
                 output_dir="./collected_papers", delay_seconds=0.1, 
                 skip_pdf_extraction=False, use_distribution=True):
        self.commercial_use = commercial_use
        self.max_papers = max_papers
        self.categories = categories or ['cs.AI', 'cs.LG']
        self.output_dir = Path(output_dir)
        self.delay_seconds = delay_seconds
        self.skip_pdf_extraction = skip_pdf_extraction
        self.use_distribution = use_distribution
        self.safe_papers = []
        self.unsafe_papers = []
        self.processed_arxiv_ids = set()
        self.session = requests.Session()
        
        # カテゴリごとの目標数を計算
        if self.use_distribution:
            self.category_targets = self._calculate_distribution_targets()
        else:
            # 均等分割
            self.category_targets = {cat: max(1, max_papers // len(self.categories)) for cat in self.categories}
    
    def _calculate_distribution_targets(self):
        """CATEGORY_DISTRIBUTIONに基づいて各カテゴリの目標論文数を計算"""
        targets = {}
        total_percentage = 0
        category_percentages = {}
        
        # 各カテゴリが属する分野とその割合を特定
        for cat in self.categories:
            for field_info in self.CATEGORY_DISTRIBUTION.values():
                if cat in field_info['categories']:
                    category_percentages[cat] = field_info['percentage'] / len(field_info['categories'])
                    total_percentage += category_percentages[cat]
                    break
            else:
                # 未定義のカテゴリは「other」として扱う
                category_percentages[cat] = 0.01  # 1%のデフォルト割合
                total_percentage += category_percentages[cat]
        
        # 正規化して目標数を計算
        if total_percentage > 0:
            for cat in self.categories:
                normalized_percentage = category_percentages.get(cat, 0) / total_percentage
                targets[cat] = max(1, int(self.max_papers * normalized_percentage))
        else:
            # フォールバック：均等分割
            targets = {cat: max(1, self.max_papers // len(self.categories)) for cat in self.categories}
        
        # 合計が max_papers を超えないよう調整
        total_target = sum(targets.values())
        if total_target > self.max_papers:
            scale_factor = self.max_papers / total_target
            for cat in targets:
                targets[cat] = max(1, int(targets[cat] * scale_factor))
        
        return targets
    
    def search_arxiv(self, category, max_results=100):
        """ArXiv APIで論文を検索"""
        url = 'http://export.arxiv.org/api/query'
        params = {
            'search_query': f'cat:{category}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'xml')
            entries = soup.find_all('entry')
            
            papers = []
            for entry in entries:
                paper = self._parse_entry(entry)
                if paper:
                    papers.append(paper)
            
            return papers
        except Exception as e:
            logger.error(f"Error searching {category}: {e}")
            return []
    
    def _parse_entry(self, entry):
        """ArXiv APIエントリをパース"""
        try:
            paper = {}
            
            # ID
            id_elem = entry.find('id')
            if id_elem:
                paper['arxiv_id'] = id_elem.text.split('/')[-1]
                paper['url'] = f"https://arxiv.org/abs/{paper['arxiv_id'].split('v')[0]}"
            
            # タイトル
            title_elem = entry.find('title')
            if title_elem:
                paper['title'] = title_elem.text.strip().replace('\n', ' ')
            
            # 著者
            authors = []
            for author in entry.find_all('author'):
                name = author.find('name')
                if name:
                    authors.append(name.text)
            paper['authors'] = ', '.join(authors)
            
            # Abstract
            summary = entry.find('summary')
            if summary:
                paper['abstract'] = summary.text.strip()
            
            # PDF URL
            paper['pdf_url'] = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf" if 'arxiv_id' in paper else None
            
            # カテゴリ
            categories = []
            for cat in entry.find_all('category'):
                if cat.get('term'):
                    categories.append(cat.get('term'))
            paper['categories'] = ', '.join(categories)
            
            # 公開日
            published = entry.find('published')
            if published:
                paper['published_date'] = published.text
            
            paper['primary_category'] = categories[0] if categories else ''
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None
    
    def check_license(self, paper):
        """論文のライセンスをチェック"""
        arxiv_url = paper.get('url', '')
        if not arxiv_url:
            return LicenseType.UNKNOWN, {'error': 'No URL'}
        
        try:
            response = self.session.get(arxiv_url, timeout=10)
            if response.status_code != 200:
                return LicenseType.UNKNOWN, {'error': f'HTTP {response.status_code}'}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # ライセンス情報を探す
            license_div = soup.find('div', class_='abs-license')
            if license_div:
                license_link = license_div.find('a', href=True)
                if license_link:
                    license_url = license_link['href']
                    license_type = self._parse_license_url(license_url)
                    
                    info = {
                        'license_url': license_url,
                        'license_text': license_div.get_text(strip=True)
                    }
                    
                    return license_type, info
            
            # ライセンスが見つからない場合はデフォルト
            return LicenseType.ARXIV_DEFAULT, {'note': 'No explicit license found'}
            
        except Exception as e:
            logger.error(f"Error checking license for {arxiv_url}: {e}")
            return LicenseType.UNKNOWN, {'error': str(e)}
    
    def _parse_license_url(self, license_url):
        """ライセンスURLからタイプを判定"""
        url_lower = license_url.lower()
        
        if 'creativecommons.org' in url_lower:
            if '/by-nc-nd/' in url_lower:
                return LicenseType.CC_BY_NC_ND
            elif '/by-nc-sa/' in url_lower:
                return LicenseType.CC_BY_NC_SA
            elif '/by-nc/' in url_lower:
                return LicenseType.CC_BY_NC
            elif '/by-nd/' in url_lower:
                return LicenseType.CC_BY_ND
            elif '/by-sa/' in url_lower:
                return LicenseType.CC_BY_SA
            elif '/by/' in url_lower:
                return LicenseType.CC_BY
            elif '/zero/' in url_lower or '/cc0/' in url_lower:
                return LicenseType.CC0
        elif 'arxiv.org/licenses/nonexclusive' in url_lower:
            return LicenseType.ARXIV_DEFAULT
        
        return LicenseType.UNKNOWN
    
    def is_safe_for_training(self, license_type):
        """LLM訓練に安全かチェック"""
        if self.commercial_use:
            return license_type in self.SAFE_LICENSES
        else:
            return license_type in (self.SAFE_LICENSES + self.RESEARCH_ONLY_LICENSES)


class PaperQualityScorer:
    """論文品質スコアリングクラス"""
    
    # 分野別の重み
    FIELD_WEIGHTS = {
        'mathematics': 1.5,
        'physics': 1.2,
        'computer_science': 1.0,
        'other': 0.8
    }
    
    @staticmethod
    def extract_theorems_and_proofs(text: str) -> Dict[str, List[str]]:
        """定理と証明を抽出（メモリ効率版）"""
        # 長いテキストは前半部分のみ処理
        if len(text) > 50000:
            text = text[:50000]
        
        results = {
            'theorems': [],
            'proofs': [],
            'lemmas': [],
            'has_rigorous_math': False
        }
        
        # シンプルなパターンマッチング（メモリ効率重視）
        theorem_pattern = r'(?:Theorem|THEOREM|Lemma|LEMMA|Proposition)\s*\d*\.?\d*[:\s]([^.]{20,300})'
        proof_pattern = r'(?:Proof|PROOF)[:\s]([^□∎]{50,500})(?:□|∎|Q\.E\.D\.|$)'
        
        theorem_matches = re.findall(theorem_pattern, text, re.IGNORECASE)[:3]
        proof_matches = re.findall(proof_pattern, text, re.IGNORECASE)[:2]
        
        results['theorems'] = [m.strip() for m in theorem_matches]
        results['proofs'] = [m.strip() for m in proof_matches]
        results['has_rigorous_math'] = len(theorem_matches) > 0 or len(proof_matches) > 0
        
        return results
    
    
    @staticmethod
    def classify_difficulty(paper: Dict) -> str:
        """論文の難易度を分類（軽量版）"""
        
        # 高度なキーワード（少数に絞る）
        advanced_keywords = [
            'homological', 'cohomology', 'spectral sequence',
            'quantum field theory', 'algebraic topology', 'homotopy'
        ]
        
        text_sample = (paper.get('title', '') + ' ' + 
                      paper.get('abstract', ''))[:3000].lower()
        
        advanced_count = sum(1 for kw in advanced_keywords if kw in text_sample)
        
        # カテゴリベース判定
        categories = paper.get('categories', '').split(', ')
        if any(cat in ['math.AG', 'math.AT', 'math.KT', 'hep-th'] for cat in categories):
            advanced_count += 2
        
        if advanced_count >= 2:
            return 'PhD_advanced'
        elif advanced_count >= 1:
            return 'PhD_intermediate'
        else:
            return 'Masters_level'
    
    @staticmethod
    def calculate_quality_score(paper: Dict, citation_data: Optional[Dict] = None) -> float:
        """論文の品質スコアを計算"""
        score = 0.0
        
        # 1. ライセンススコア（安全なライセンスを優先）
        license_type = paper.get('license_type', '')
        if license_type in ['CC BY', 'CC0']:
            score += 20
        elif license_type == 'CC BY-SA':
            score += 15
        elif license_type in ['ArXiv Non-exclusive', 'CC BY-NC', 'CC BY-NC-SA']:
            score += 10
        
        # 2. 数学的厳密性スコア
        content = paper.get('content', '')[:20000]  # 最初の20000文字のみ
        if content:
            math_indicators = ['theorem', 'proof', 'lemma', 'proposition', 'corollary']
            math_count = sum(content.lower().count(ind) for ind in math_indicators)
            score += min(20, math_count * 2)
        
        # 3. 引用数スコア（利用可能な場合）
        if citation_data:
            citations = citation_data.get('citation_count', 0)
            if citations > 100:
                score += 25
            elif citations > 50:
                score += 20
            elif citations > 20:
                score += 15
            elif citations > 5:
                score += 10
            else:
                score += 5
        
        # 4. 論文の新しさ（3年以内を優先）
        try:
            pub_date = paper.get('published_date', '')
            if pub_date:
                pub_year = int(pub_date[:4])
                years_old = 2025 - pub_year
                if years_old <= 1:
                    score += 15
                elif years_old <= 3:
                    score += 10
                elif years_old <= 5:
                    score += 5
        except:
            pass
        
        # 5. 分野による重み付け
        primary_cat = paper.get('primary_category', '')
        field = 'other'
        for field_name, field_info in ArxivScraper.CATEGORY_DISTRIBUTION.items():
            if primary_cat in field_info['categories']:
                field = field_name
                break
        
        field_weight = PaperQualityScorer.FIELD_WEIGHTS.get(field, 1.0)
        score *= field_weight
        
        return min(100, score)  # 最大100点


class SemanticScholarAPI:
    """Semantic Scholar APIクライアント（軽量版）"""
    
    def __init__(self, cache_dir: str = "./citation_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ArXiv-PhD-Scraper/1.0'
        })
    
    def get_citation_count(self, arxiv_id: str, use_cache: bool = True) -> Optional[Dict]:
        """引用数を取得（キャッシュ付き）"""
        
        # キャッシュチェック
        cache_file = self.cache_dir / f"{arxiv_id.replace('/', '_')}_citations.json"
        
        if use_cache and cache_file.exists():
            try:
                # 7日以内のキャッシュのみ使用
                if (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)) < timedelta(days=7):
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            except:
                pass
        
        # API呼び出し
        try:
            # ArXiv IDの整形
            clean_id = arxiv_id.split('v')[0]
            
            # Semantic Scholar API（無料枠）
            url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_id}"
            params = {'fields': 'citationCount,year,influentialCitationCount'}
            
            response = self.session.get(url, params=params, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                result = {
                    'citation_count': data.get('citationCount', 0),
                    'influential_citations': data.get('influentialCitationCount', 0),
                    'year': data.get('year'),
                    'fetched_at': datetime.now().isoformat()
                }
                
                # キャッシュ保存
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
                
                return result
                
        except Exception as e:
            logger.debug(f"Citation fetch failed for {arxiv_id}: {e}")
        
        return None


class EnhancedArxivScraper(ArxivScraper):
    """拡張版ArXivスクレイパー"""
    
    def __init__(self, commercial_use=False, max_papers=100, categories=None,
                 output_dir="./collected_papers", delay_seconds=0.1,
                 skip_pdf_extraction=False, min_quality_score=30,
                 use_citations=True, prefer_recent=True, max_pdf_pages=15,
                 use_distribution=True, enable_rag_chunking=True,
                 max_chunk_size=2000, chunk_overlap=200, session_timestamp=None):
        # 新しいパラメータ
        self.min_quality_score = min_quality_score
        self.use_citations = use_citations
        self.prefer_recent = prefer_recent
        self.max_pdf_pages = max_pdf_pages
        self.enable_rag_chunking = enable_rag_chunking
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.session_timestamp = session_timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 親クラス初期化
        super().__init__(
            commercial_use=commercial_use,
            max_papers=max_papers,
            categories=categories,
            output_dir=output_dir,
            delay_seconds=delay_seconds,
            skip_pdf_extraction=skip_pdf_extraction,
            use_distribution=use_distribution
        )
        
        # 追加コンポーネント
        self.quality_scorer = PaperQualityScorer()
        self.citation_api = SemanticScholarAPI() if self.use_citations else None
        
        # RAG用セクション分割器
        if self.enable_rag_chunking:
            self.section_splitter = PaperSectionSplitter(
                max_chunk_size=max_chunk_size,
                overlap_size=chunk_overlap
            )
        
        # メモリ管理
        self.papers_processed = 0
        self.gc_interval = 20  # 20論文ごとにガベージコレクション
    
    def extract_pdf_content_enhanced(self, pdf_url: str) -> Dict:
        """拡張版PDF内容抽出（メモリ効率的）"""
        
        try:
            # PDFダウンロード（タイムアウト短縮）
            response = self.session.get(pdf_url, timeout=15)
            if response.status_code != 200:
                return {}
            
            # ファイルサイズチェック（10MB以上はスキップ）
            if len(response.content) > 10 * 1024 * 1024:
                logger.info("  PDF too large, skipping detailed extraction")
                return {'content': '', 'extraction_note': 'PDF_too_large'}
            
            # PDF読み込み
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # ページ数制限
            pages_to_read = min(len(pdf_reader.pages), self.max_pdf_pages)
            
            # テキスト抽出
            text_parts = []
            for page_num in range(pages_to_read):
                try:
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except:
                    continue
            
            full_text = '\n'.join(text_parts)
            
            # メモリ効率のため長すぎるテキストは切り詰める
            if len(full_text) > 50000:
                full_text = full_text[:50000]
            
            # 定理・証明を抽出
            theorems_proofs = self.quality_scorer.extract_theorems_and_proofs(full_text)
            
            # 重要な数式の抽出をスキップ（高速化）
            formulas = []
            
            result = {
                'content': full_text[:20000],  # 基本コンテンツ（20000文字まで）
                'has_theorems': len(theorems_proofs['theorems']) > 0,
                'has_proofs': len(theorems_proofs['proofs']) > 0,
                'theorem_count': len(theorems_proofs['theorems']),
                'proof_count': len(theorems_proofs['proofs']),
                'formula_count': len(formulas),
                'sample_theorem': theorems_proofs['theorems'][0][:200] if theorems_proofs['theorems'] else '',
                'sample_proof': theorems_proofs['proofs'][0][:200] if theorems_proofs['proofs'] else '',
                'sample_formulas': ''
            }
            
            # メモリ解放
            del pdf_file, pdf_reader, text_parts
            
            return result
            
        except Exception as e:
            logger.debug(f"Error in enhanced PDF extraction: {e}")
            return {'content': '', 'extraction_error': str(e)}
    
    def evaluate_paper(self, paper: Dict) -> Dict:
        """論文を評価してメタデータを追加"""
        
        # 引用数を取得（オプション）
        citation_data = None
        if self.use_citations and self.citation_api:
            arxiv_id = paper.get('arxiv_id', '')
            if arxiv_id:
                citation_data = self.citation_api.get_citation_count(arxiv_id)
                if citation_data:
                    paper['citation_count'] = citation_data.get('citation_count', 0)
                    paper['influential_citations'] = citation_data.get('influential_citations', 0)
        
        # 品質スコア計算
        quality_score = self.quality_scorer.calculate_quality_score(paper, citation_data)
        paper['quality_score'] = quality_score
        
        # 難易度分類
        difficulty = self.quality_scorer.classify_difficulty(paper)
        paper['difficulty_level'] = difficulty
        
        # データセット生成への適合性
        paper['dataset_suitability'] = {
            'has_proofs': paper.get('has_proofs', False),
            'has_theorems': paper.get('has_theorems', False),
            'theorem_count': paper.get('theorem_count', 0),
            'proof_count': paper.get('proof_count', 0),
            'formula_count': paper.get('formula_count', 0),
            'suitable_for_training': quality_score >= self.min_quality_score
        }
        
        return paper
    
    def scrape_safe_papers(self) -> Tuple[List[Dict], List[Dict]]:
        """品質重視の論文収集"""
        
        logger.info("\n" + "="*60)
        logger.info("🎓 PhD-Level ArXiv Paper Collection")
        logger.info("="*60)
        logger.info(f"Mode: {'Commercial' if self.commercial_use else 'Research'}")
        logger.info(f"Target: {self.max_papers} high-quality papers")
        logger.info(f"Min quality score: {self.min_quality_score}")
        logger.info(f"Using citations: {self.use_citations}")
        
        # カテゴリごとの目標を表示
        if hasattr(self, 'category_targets'):
            logger.info("\nCategory targets:")
            for cat, target in self.category_targets.items():
                logger.info(f"  {cat}: {target} papers")
        
        # カテゴリごとの収集
        category_collected = {cat: 0 for cat in self.categories}
        high_quality_papers = []
        
        for category in self.categories:
            if len(high_quality_papers) >= self.max_papers:
                break
            
            target_for_category = self.category_targets.get(category, 1)
            
            logger.info(f"\nSearching {category} (target: {target_for_category})...")
            
            # 多めに検索（フィルタリングで減るため）
            search_limit = min(100, target_for_category * 10)
            papers = self.search_arxiv(category, search_limit)
            
            # 品質でソート
            evaluated_papers = []
            for paper in papers:
                if paper.get('arxiv_id') in self.processed_arxiv_ids:
                    continue
                
                # ライセンスチェック
                license_type, license_info = self.check_license(paper)
                paper['license_type'] = license_type.value
                paper['license_info'] = license_info
                
                if not self.is_safe_for_training(license_type):
                    self.unsafe_papers.append(paper)
                    continue
                
                # PDF内容抽出（拡張版）
                if not self.skip_pdf_extraction and paper.get('pdf_url'):
                    logger.info(f"  Extracting: {paper.get('title', '')[:50]}...")
                    pdf_data = self.extract_pdf_content_enhanced(paper['pdf_url'])
                    paper.update(pdf_data)
                
                # 論文評価
                paper = self.evaluate_paper(paper)
                
                # 品質フィルタ
                if paper['quality_score'] >= self.min_quality_score:
                    evaluated_papers.append(paper)
                    logger.info(f"  ✅ Quality: {paper['quality_score']:.1f}, "
                              f"Difficulty: {paper['difficulty_level']}, "
                              f"Citations: {paper.get('citation_count', 'N/A')}")
                
                # メモリ管理
                self.papers_processed += 1
                if self.papers_processed % self.gc_interval == 0:
                    gc.collect()
                
                time.sleep(self.delay_seconds)
                
                # 十分な数が集まったら次のカテゴリへ
                if len(evaluated_papers) >= target_for_category:
                    break
            
            # 品質スコアでソート
            evaluated_papers.sort(key=lambda x: x['quality_score'], reverse=True)
            
            # 目標数まで追加
            for paper in evaluated_papers[:target_for_category]:
                if len(high_quality_papers) >= self.max_papers:
                    break
                high_quality_papers.append(paper)
                category_collected[category] += 1
                self.processed_arxiv_ids.add(paper.get('arxiv_id', ''))
        
        self.safe_papers = high_quality_papers
        
        # 結果サマリー
        logger.info("\n" + "="*60)
        logger.info(f"✅ Collected {len(self.safe_papers)} high-quality papers")
        logger.info(f"❌ Rejected {len(self.unsafe_papers)} papers")
        
        # カテゴリ別の収集結果を表示
        if category_collected:
            logger.info("\n📊 Papers by category:")
            for cat, count in category_collected.items():
                target = self.category_targets.get(cat, 0)
                percentage = (count / target * 100) if target > 0 else 0
                logger.info(f"  {cat}: {count}/{target} ({percentage:.0f}% of target)")
        
        # 品質統計
        if self.safe_papers:
            avg_quality = sum(p['quality_score'] for p in self.safe_papers) / len(self.safe_papers)
            phd_advanced = sum(1 for p in self.safe_papers if p.get('difficulty_level') == 'PhD_advanced')
            phd_intermediate = sum(1 for p in self.safe_papers if p.get('difficulty_level') == 'PhD_intermediate')
            
            logger.info(f"\n📊 Quality Statistics:")
            logger.info(f"  Average quality score: {avg_quality:.1f}")
            logger.info(f"  PhD Advanced: {phd_advanced}")
            logger.info(f"  PhD Intermediate: {phd_intermediate}")
            
            papers_with_proofs = sum(1 for p in self.safe_papers if p.get('has_proofs'))
            papers_with_theorems = sum(1 for p in self.safe_papers if p.get('has_theorems'))
            logger.info(f"  Papers with proofs: {papers_with_proofs}")
            logger.info(f"  Papers with theorems: {papers_with_theorems}")
        
        return self.safe_papers, self.unsafe_papers
    
    def save_results(self, append_mode=False) -> Dict[str, str]:
        """結果をJSONL形式で保存（RAGチャンクも含む）"""
        
        timestamp = self.session_timestamp
        files = {}
        
        # 出力ディレクトリが存在しない場合は作成
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 安全な論文をJSONL形式で保存
        if self.safe_papers:
            safe_file = self.output_dir / f"safe_papers_{timestamp}.jsonl"
            mode = 'a' if append_mode else 'w'
            with open(safe_file, mode, encoding='utf-8') as f:
                for paper in self.safe_papers:
                    # 各論文を1行のJSONとして保存
                    json_line = json.dumps(paper, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            
            files['safe'] = str(safe_file)
            action = "Appended" if append_mode else "Saved"
            logger.info(f"✅ {action} {len(self.safe_papers)} safe papers to {safe_file}")
            
            # RAGチャンクを生成して保存
            if self.enable_rag_chunking and hasattr(self, 'section_splitter'):
                rag_dir = self.output_dir / "rag_data"
                rag_dir.mkdir(exist_ok=True, parents=True)
                rag_file = rag_dir / f"rag_chunks_{timestamp}.jsonl"
                
                total_chunks = 0
                chunks_by_type = {}
                
                mode = 'a' if append_mode else 'w'
                with open(rag_file, mode, encoding='utf-8') as f:
                    for paper in self.safe_papers:
                        # 各論文をセクション分割
                        chunks = self.section_splitter.process_paper(paper)
                        
                        for chunk in chunks:
                            # チャンクIDを生成
                            chunk['chunk_id'] = f"{chunk['arxiv_id']}_{chunk['section_index']}_{chunk['chunk_index']}"
                            
                            # JSONLとして書き込み
                            json.dump(chunk, f, ensure_ascii=False)
                            f.write('\n')
                            
                            # 統計更新
                            total_chunks += 1
                            section_type = chunk['section_type']
                            chunks_by_type[section_type] = chunks_by_type.get(section_type, 0) + 1
                
                files['rag_chunks'] = str(rag_file)
                action = "Appended" if append_mode else "Saved"
                logger.info(f"📚 {action} {total_chunks} RAG chunks to {rag_file}")
                
                # RAG統計を更新（append時は既存の統計に追加）
                rag_stats_file = rag_dir / f"rag_stats_{timestamp}.json"
                
                existing_stats = {'total_chunks': 0, 'chunks_by_type': {}, 'papers_processed': 0}
                if append_mode and rag_stats_file.exists():
                    with open(rag_stats_file, 'r', encoding='utf-8') as f:
                        existing_stats = json.load(f)
                
                # 統計を更新
                existing_stats['total_chunks'] += total_chunks
                existing_stats['papers_processed'] += len(self.safe_papers)
                for k, v in chunks_by_type.items():
                    existing_stats['chunks_by_type'][k] = existing_stats['chunks_by_type'].get(k, 0) + v
                existing_stats['timestamp'] = timestamp
                existing_stats['last_updated'] = datetime.now().isoformat()
                
                with open(rag_stats_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_stats, f, indent=2, ensure_ascii=False)
                
                files['rag_stats'] = str(rag_stats_file)
                logger.info(f"📊 Updated RAG statistics in {rag_stats_file}")
        
        # 拒否された論文もJSONL形式で保存（参考用）
        if self.unsafe_papers:
            unsafe_file = self.output_dir / f"unsafe_papers_{timestamp}.jsonl"
            mode = 'a' if append_mode else 'w'
            with open(unsafe_file, mode, encoding='utf-8') as f:
                for paper in self.unsafe_papers:
                    json_line = json.dumps(paper, ensure_ascii=False)
                    f.write(json_line + '\n')
            
            
            files['unsafe'] = str(unsafe_file)
            action = "Appended" if append_mode else "Saved"
            logger.info(f"📝 {action} {len(self.unsafe_papers)} unsafe papers to {unsafe_file}")
        
        # メタデータファイルを更新（append時は累積）
        metadata_file = self.output_dir / f"metadata_{timestamp}.json"
        
        existing_metadata = {}
        if append_mode and metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
        
        # 現在のバッチのメタデータ
        current_safe = len(self.safe_papers)
        current_unsafe = len(self.unsafe_papers)
        
        metadata = {
            'timestamp': timestamp,
            'session_start': existing_metadata.get('session_start', datetime.now().isoformat()),
            'last_updated': datetime.now().isoformat(),
            'total_safe_papers': existing_metadata.get('total_safe_papers', 0) + current_safe,
            'total_unsafe_papers': existing_metadata.get('total_unsafe_papers', 0) + current_unsafe,
            'categories_searched': list(self.categories) if hasattr(self, 'categories') else [],
            'commercial_use': self.commercial_use if hasattr(self, 'commercial_use') else False,
            'quality_stats': {}
        }
        
        # 品質統計を追加（利用可能な場合）
        if self.safe_papers and any('quality_score' in p for p in self.safe_papers):
            quality_scores = [p.get('quality_score', 0) for p in self.safe_papers if 'quality_score' in p]
            if quality_scores:
                metadata['quality_stats'] = {
                    'average_quality_score': sum(quality_scores) / len(quality_scores),
                    'max_quality_score': max(quality_scores),
                    'min_quality_score': min(quality_scores),
                    'papers_with_proofs': sum(1 for p in self.safe_papers if p.get('has_proofs')),
                    'papers_with_theorems': sum(1 for p in self.safe_papers if p.get('has_theorems')),
                    'papers_with_citations': sum(1 for p in self.safe_papers if p.get('citation_count', 0) > 0)
                }
            
            # 難易度分布
            difficulty_dist = {}
            for paper in self.safe_papers:
                level = paper.get('difficulty_level', 'Unknown')
                difficulty_dist[level] = difficulty_dist.get(level, 0) + 1
            metadata['difficulty_distribution'] = difficulty_dist
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        
        files['metadata'] = str(metadata_file)
        logger.info(f"📋 Updated metadata in {metadata_file}")
        
        # サマリーレポート更新（append時は追記）
        report_file = self.output_dir / f"scraping_report_{timestamp}.txt"
        report = self.generate_report()
        mode = 'a' if append_mode else 'w'
        
        if append_mode:
            report = f"\n\n{'='*60}\nBatch Update at {datetime.now().isoformat()}\n{'='*60}\n" + report
        
        with open(report_file, mode, encoding='utf-8') as f:
            f.write(report)
        
        
        files['report'] = str(report_file)
        action = "Updated" if append_mode else "Saved"
        logger.info(f"📊 {action} report in {report_file}")
        
        return files
    
    def generate_report(self) -> str:
        """詳細レポート生成"""
        
        report = []
        report.append("="*60)
        report.append("🎓 PhD-Level ArXiv Paper Collection Report")
        report.append("="*60)
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        report.append(f"Total safe papers: {len(self.safe_papers)}")
        report.append(f"Total unsafe papers: {len(self.unsafe_papers)}")
        
        if self.safe_papers:
            # 品質統計
            if any('quality_score' in p for p in self.safe_papers):
                quality_scores = [p.get('quality_score', 0) for p in self.safe_papers if 'quality_score' in p]
                if quality_scores:
                    report.append(f"\n📊 Quality Statistics:")
                    report.append(f"  Average quality score: {sum(quality_scores)/len(quality_scores):.1f}")
                    report.append(f"  Max quality score: {max(quality_scores):.1f}")
                    report.append(f"  Min quality score: {min(quality_scores):.1f}")
            
            # 定理・証明統計
            papers_with_proofs = sum(1 for p in self.safe_papers if p.get('has_proofs'))
            papers_with_theorems = sum(1 for p in self.safe_papers if p.get('has_theorems'))
            if papers_with_proofs or papers_with_theorems:
                report.append(f"\n📚 Mathematical Content:")
                report.append(f"  Papers with proofs: {papers_with_proofs}")
                report.append(f"  Papers with theorems: {papers_with_theorems}")
            
            # 引用統計
            papers_with_citations = [p for p in self.safe_papers if p.get('citation_count', 0) > 0]
            if papers_with_citations:
                citations = [p['citation_count'] for p in papers_with_citations]
                report.append(f"\n📈 Citation Statistics:")
                report.append(f"  Papers with citations: {len(papers_with_citations)}")
                report.append(f"  Average citations: {sum(citations)/len(citations):.1f}")
                report.append(f"  Max citations: {max(citations)}")
            
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


class PaperSectionSplitter:
    """論文をセクション単位で分割"""
    
    # 一般的な論文セクションパターン
    SECTION_PATTERNS = [
        r'^(?:\d+\.?\s*)?(?:Abstract|ABSTRACT)',
        r'^(?:\d+\.?\s*)?(?:Introduction|INTRODUCTION)',
        r'^(?:\d+\.?\s*)?(?:Related Work|RELATED WORK|Literature Review)',
        r'^(?:\d+\.?\s*)?(?:Background|BACKGROUND)',
        r'^(?:\d+\.?\s*)?(?:Method|Methods|Methodology|METHODOLOGY)',
        r'^(?:\d+\.?\s*)?(?:Approach|APPROACH|Proposed Method)',
        r'^(?:\d+\.?\s*)?(?:Experiments?|EXPERIMENTS?|Experimental Setup)',
        r'^(?:\d+\.?\s*)?(?:Results?|RESULTS?)',
        r'^(?:\d+\.?\s*)?(?:Discussion|DISCUSSION)',
        r'^(?:\d+\.?\s*)?(?:Conclusion|CONCLUSION|Conclusions)',
        r'^(?:\d+\.?\s*)?(?:References|REFERENCES|Bibliography)',
        r'^(?:\d+\.?\s*)?(?:Appendix|APPENDIX)',
    ]
    
    # セクションタイプのマッピング
    SECTION_TYPES = {
        'abstract': ['abstract'],
        'introduction': ['introduction'],
        'related_work': ['related work', 'literature review'],
        'background': ['background'],
        'methodology': ['method', 'methods', 'methodology', 'approach', 'proposed method'],
        'experiments': ['experiment', 'experiments', 'experimental setup'],
        'results': ['result', 'results'],
        'discussion': ['discussion'],
        'conclusion': ['conclusion', 'conclusions'],
        'references': ['references', 'bibliography'],
        'appendix': ['appendix']
    }
    
    def __init__(self, max_chunk_size: int = 2000, overlap_size: int = 200):
        """
        Args:
            max_chunk_size: 各チャンクの最大文字数
            overlap_size: チャンク間のオーバーラップサイズ
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.compiled_patterns = [re.compile(p, re.MULTILINE | re.IGNORECASE) 
                                  for p in self.SECTION_PATTERNS]
    
    def identify_section_type(self, section_title: str) -> str:
        """セクションタイトルからタイプを識別"""
        title_lower = section_title.lower().strip()
        
        for section_type, keywords in self.SECTION_TYPES.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return section_type
        
        return 'other'
    
    def extract_sections(self, text: str) -> List[Dict]:
        """テキストからセクションを抽出"""
        sections = []
        
        # セクションの開始位置を検出
        section_starts = []
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                section_starts.append({
                    'position': match.start(),
                    'title': match.group().strip(),
                    'end_of_title': match.end()
                })
        
        # 位置でソート
        section_starts.sort(key=lambda x: x['position'])
        
        # 各セクションのコンテンツを抽出
        for i, section in enumerate(section_starts):
            start_pos = section['end_of_title']
            end_pos = section_starts[i + 1]['position'] if i + 1 < len(section_starts) else len(text)
            
            content = text[start_pos:end_pos].strip()
            
            if content:  # 空のセクションはスキップ
                section_type = self.identify_section_type(section['title'])
                sections.append({
                    'section_title': section['title'],
                    'section_type': section_type,
                    'content': content,
                    'section_index': i,
                    'char_count': len(content)
                })
        
        # セクションが見つからない場合は全体を1つのセクションとして扱う
        if not sections and text.strip():
            sections.append({
                'section_title': 'Full Text',
                'section_type': 'full_text',
                'content': text.strip(),
                'section_index': 0,
                'char_count': len(text.strip())
            })
        
        return sections
    
    def split_large_section(self, section: Dict) -> List[Dict]:
        """大きなセクションをチャンクに分割"""
        content = section['content']
        
        if len(content) <= self.max_chunk_size:
            return [section]
        
        chunks = []
        sentences = self._split_into_sentences(content)
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # 現在のチャンクを保存
                chunk_content = ' '.join(current_chunk)
                chunks.append({
                    'section_title': f"{section['section_title']} (Part {chunk_index + 1})",
                    'section_type': section['section_type'],
                    'content': chunk_content,
                    'section_index': section['section_index'],
                    'chunk_index': chunk_index,
                    'is_chunked': True,
                    'char_count': len(chunk_content)
                })
                
                # オーバーラップを考慮して次のチャンクを開始
                if self.overlap_size > 0:
                    overlap_sentences = []
                    overlap_size = 0
                    for sent in reversed(current_chunk):
                        overlap_size += len(sent)
                        overlap_sentences.insert(0, sent)
                        if overlap_size >= self.overlap_size:
                            break
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
                
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # 最後のチャンクを保存
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append({
                'section_title': f"{section['section_title']} (Part {chunk_index + 1})" if chunk_index > 0 else section['section_title'],
                'section_type': section['section_type'],
                'content': chunk_content,
                'section_index': section['section_index'],
                'chunk_index': chunk_index,
                'is_chunked': chunk_index > 0,
                'char_count': len(chunk_content)
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """テキストを文に分割（簡易版）"""
        # 改良された文分割パターン
        sentence_endings = re.compile(r'[.!?]+[\s\n]+')
        sentences = sentence_endings.split(text)
        
        # 空の文を除去
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 文が長すぎる場合は段落で分割
        final_sentences = []
        for sentence in sentences:
            if len(sentence) > self.max_chunk_size // 2:
                # 段落で分割
                paragraphs = sentence.split('\n\n')
                final_sentences.extend(paragraphs)
            else:
                final_sentences.append(sentence)
        
        return final_sentences
    
    def process_paper(self, paper_data: Dict) -> List[Dict]:
        """論文データを処理してセクション化されたチャンクを生成"""
        
        # テキストコンテンツを取得
        full_text = paper_data.get('content', '')
        if not full_text:
            full_text = paper_data.get('abstract', '')
        
        if not full_text:
            return []
        
        # セクションを抽出
        sections = self.extract_sections(full_text)
        
        # 各セクションを適切なサイズに分割
        all_chunks = []
        for section in sections:
            chunks = self.split_large_section(section)
            
            # 各チャンクにメタデータを追加
            for chunk in chunks:
                chunk_data = {
                    'arxiv_id': paper_data.get('arxiv_id'),
                    'paper_title': paper_data.get('title'),
                    'authors': paper_data.get('authors'),
                    'categories': paper_data.get('categories'),
                    'published_date': paper_data.get('published_date'),
                    'section_title': chunk['section_title'],
                    'section_type': chunk['section_type'],
                    'section_index': chunk['section_index'],
                    'chunk_index': chunk.get('chunk_index', 0),
                    'is_chunked': chunk.get('is_chunked', False),
                    'content': chunk['content'],
                    'char_count': chunk['char_count'],
                    'embedding_ready': True  # 埋め込みベクトル生成準備完了フラグ
                }
                
                # 品質スコアがある場合は追加
                if 'quality_score' in paper_data:
                    chunk_data['paper_quality_score'] = paper_data['quality_score']
                
                # ライセンス情報
                if 'license_type' in paper_data:
                    chunk_data['license_type'] = paper_data['license_type']
                
                all_chunks.append(chunk_data)
        
        return all_chunks


class CheckpointManager:
    """チェックポイント管理クラス"""
    
    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_file = self.checkpoint_dir / "arxiv_scraper_checkpoint.pkl"
        self.backup_file = self.checkpoint_dir / "arxiv_scraper_checkpoint_backup.pkl"
    
    def save_checkpoint(self, state: Dict) -> bool:
        """チェックポイントを保存"""
        try:
            # 既存のチェックポイントをバックアップ
            if self.checkpoint_file.exists():
                import shutil
                shutil.copy2(self.checkpoint_file, self.backup_file)
            
            # 新しいチェックポイントを保存
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"💾 Checkpoint saved: {self.checkpoint_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self) -> Optional[Dict]:
        """チェックポイントを読み込み"""
        try:
            if not self.checkpoint_file.exists():
                return None
            
            with open(self.checkpoint_file, 'rb') as f:
                state = pickle.load(f)
            
            logger.info(f"📂 Checkpoint loaded from: {self.checkpoint_file}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            
            # バックアップから復元を試みる
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, 'rb') as f:
                        state = pickle.load(f)
                    logger.info(f"📂 Checkpoint loaded from backup")
                    return state
                except:
                    pass
            
            return None
    
    def clear_checkpoint(self):
        """チェックポイントをクリア"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.backup_file.exists():
                self.backup_file.unlink()
            logger.info("🗑️ Checkpoints cleared")
        except Exception as e:
            logger.error(f"Failed to clear checkpoints: {e}")


def main():
    """連続実行版メイン関数（中断・再開機能付き）"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced ArXiv scraper for PhD datasets")
    parser.add_argument("--papers-per-batch", type=int, default=10,
                       help="Number of papers to collect per batch")
    parser.add_argument("--categories", nargs="+", 
                       default=['cs.AI', 'cs.LG', 'math.CO', 'physics.comp-ph'],
                       help="ArXiv categories to search")
    parser.add_argument("--min-quality", type=float, default=30,
                       help="Minimum quality score (0-100)")
    parser.add_argument("--use-citations", action="store_true", default=True,
                       help="Fetch citation counts")
    parser.add_argument("--output-dir", type=str, default="/home/Competition2025/P05/shareP05/data_generation/arxiv_papers",
                       help="Output directory")
    parser.add_argument("--batch-interval", type=int, default=300,
                       help="Seconds to wait between batches (default: 5 minutes)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    parser.add_argument("--clear-checkpoint", action="store_true",
                       help="Clear existing checkpoint and start fresh")
    parser.add_argument("--save-interval", type=int, default=5,
                       help="Save checkpoint every N batches")
    
    # RAG関連のオプション
    parser.add_argument("--enable-rag-chunking", action="store_true", default=True,
                       help="Enable RAG-optimized chunking of papers")
    parser.add_argument("--max-chunk-size", type=int, default=2000,
                       help="Maximum chunk size in characters for RAG")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                       help="Overlap size between chunks")
    
    args = parser.parse_args()
    
    # チェックポイントマネージャー初期化
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)
    
    # チェックポイントクリアオプション
    if args.clear_checkpoint:
        checkpoint_mgr.clear_checkpoint()
        logger.info("Starting fresh collection...\n")
    
    # Ctrl+Cで停止するためのフラグ
    running = True
    total_collected = 0
    batch_number = 0
    all_processed_ids = set()
    start_time = datetime.now()
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # セッション全体で共通のタイムスタンプ
    
    # チェックポイントから再開
    if args.resume:
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            total_collected = checkpoint.get('total_collected', 0)
            batch_number = checkpoint.get('batch_number', 0)
            all_processed_ids = set(checkpoint.get('processed_ids', []))
            start_time = checkpoint.get('start_time', datetime.now())
            
            logger.info("="*60)
            logger.info("📥 Resuming from checkpoint:")
            logger.info(f"  Last batch: #{batch_number}")
            logger.info(f"  Papers collected: {total_collected}")
            logger.info(f"  Processed IDs: {len(all_processed_ids)}")
            logger.info(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("="*60)
        else:
            logger.info("No checkpoint found, starting fresh...\n")
    
    def signal_handler(_signum, _frame):
        nonlocal running
        logger.info("\n⚠️  Stopping collection gracefully...")
        logger.info("💾 Saving final checkpoint...")
        
        # 最終チェックポイントを保存
        checkpoint_state = {
            'total_collected': total_collected,
            'batch_number': batch_number,
            'processed_ids': list(all_processed_ids),
            'start_time': start_time,
            'args': vars(args),
            'stopped_at': datetime.now()
        }
        checkpoint_mgr.save_checkpoint(checkpoint_state)
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("="*60)
    logger.info("🔄 Continuous ArXiv Paper Collection")
    logger.info("="*60)
    logger.info(f"Papers per batch: {args.papers_per_batch}")
    logger.info(f"Batch interval: {args.batch_interval} seconds")
    logger.info(f"Categories: {', '.join(args.categories)}")
    logger.info(f"Checkpoint save interval: every {args.save_interval} batches")
    logger.info("Press Ctrl+C to stop collection gracefully\n")
    
    while running:
        batch_number += 1
        logger.info(f"\n📦 Starting batch #{batch_number}")
        logger.info("="*40)
        
        try:
            # 拡張版スクレイパー初期化（各バッチで新規）
            scraper = EnhancedArxivScraper(
                commercial_use=False,
                max_papers=args.papers_per_batch,
                categories=args.categories,
                min_quality_score=args.min_quality,
                use_citations=args.use_citations,
                output_dir=args.output_dir,
                delay_seconds=0.5,  # API制限回避
                skip_pdf_extraction=False,
                max_pdf_pages=10,  # メモリ節約
                enable_rag_chunking=args.enable_rag_chunking,
                max_chunk_size=args.max_chunk_size,
                chunk_overlap=args.chunk_overlap,
                session_timestamp=session_timestamp  # セッション共通のタイムスタンプを使用
            )
            
            # 既に処理済みのIDを設定
            scraper.processed_arxiv_ids = all_processed_ids.copy()
            
            # 論文収集
            safe_papers, _ = scraper.scrape_safe_papers()
            
            # 結果保存（2回目以降はappendモード）
            if safe_papers:
                append_mode = batch_number > 1
                files = scraper.save_results(append_mode=append_mode)
                total_collected += len(safe_papers)
                
                # 処理済みIDを更新
                for paper in safe_papers:
                    if 'arxiv_id' in paper:
                        all_processed_ids.add(paper['arxiv_id'])
                
                logger.info(f"\n📊 Batch #{batch_number} Complete:")
                logger.info(f"  Papers collected: {len(safe_papers)}")
                logger.info(f"  Total collected: {total_collected}")
                logger.info(f"  Saved to: {files.get('safe', 'N/A')}")
                
                # 定期的にチェックポイントを保存
                if batch_number % args.save_interval == 0:
                    checkpoint_state = {
                        'total_collected': total_collected,
                        'batch_number': batch_number,
                        'processed_ids': list(all_processed_ids),
                        'start_time': start_time,
                        'args': vars(args),
                        'last_saved': datetime.now()
                    }
                    checkpoint_mgr.save_checkpoint(checkpoint_state)
            else:
                logger.info(f"\n⚠️  No new papers found in batch #{batch_number}")
            
            # 次のバッチまで待機
            if running:
                logger.info(f"\n⏰ Waiting {args.batch_interval} seconds until next batch...")
                for i in range(args.batch_interval):
                    if not running:
                        break
                    time.sleep(1)
                    if i % 60 == 0 and i > 0:
                        remaining = args.batch_interval - i
                        logger.info(f"   {remaining} seconds remaining...")
        
        except Exception as e:
            logger.error(f"\n❌ Error in batch #{batch_number}: {e}")
            logger.info("Continuing to next batch...")
            time.sleep(30)  # エラー時は短めの待機
    
    # 最終統計とチェックポイント保存
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "="*60)
    logger.info("📊 Final Statistics:")
    logger.info(f"  Total batches: {batch_number}")
    logger.info(f"  Total papers collected: {total_collected}")
    logger.info(f"  Total runtime: {duration}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    if total_collected > 0:
        avg_time_per_paper = duration.total_seconds() / total_collected
        logger.info(f"  Average time per paper: {avg_time_per_paper:.1f} seconds")
    
    # 正常終了時のチェックポイント保存
    if not args.clear_checkpoint:
        final_state = {
            'total_collected': total_collected,
            'batch_number': batch_number,
            'processed_ids': list(all_processed_ids),
            'start_time': start_time,
            'end_time': end_time,
            'args': vars(args),
            'completed': True
        }
        checkpoint_mgr.save_checkpoint(final_state)
        logger.info(f"\n💾 Final checkpoint saved for future reference")
        logger.info(f"   Use --resume to continue collecting more papers")
        logger.info(f"   Use --clear-checkpoint to start fresh")
    
    logger.info("="*60)
    
    return total_collected


if __name__ == "__main__":
    total_papers = main()