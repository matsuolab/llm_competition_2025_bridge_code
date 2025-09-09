import logging
import multiprocessing
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DataProcessor:
    """様々な形式のデータセットを統一的に処理するクラス"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # CPUコア数を取得（最大240コアまで）
        self.num_proc = min(multiprocessing.cpu_count(), 240)
        logger.info(f"DataProcessor will use {self.num_proc} CPU cores for parallel processing")
    
    def process_dataset(self, dataset):
        """データセットの形式を自動判定して処理"""
        column_names = dataset.column_names
        logger.info(f"Dataset columns: {column_names}")
        logger.info(f"Dataset size: {len(dataset)} examples")
        
        # 各形式の処理メソッドを定義
        processors = [
            (["messages"], self._process_messages_format),
            (["question", "answer"], self._process_qa_format),
            (["input", "output"], self._process_io_format),
            (["text"], self._process_text_format),
            (["prompt", "completion"], self._process_pc_format),
            (["instruction", "response"], self._process_ir_format),
        ]
        
        # 適切な処理メソッドを選択
        for required_cols, processor in processors:
            if all(col in column_names for col in required_cols):
                logger.info(f"Using processor for columns: {required_cols}")
                return processor(dataset, column_names)
        
        # 対応していない形式の場合
        raise ValueError(
            f"Unsupported dataset format. Found columns: {column_names}\n"
            "Supported formats:\n"
            "  - messages (chat format)\n"
            "  - question/answer\n"
            "  - input/output\n"
            "  - prompt/completion\n"
            "  - instruction/response (with optional input)\n"
            "  - text (pre-formatted)"
        )
    
    def _process_messages_format(self, dataset, column_names):
        """messagesカラムの処理（チャット形式）"""
        logger.info("Processing chat format dataset (messages column)")
        
        def format_chat_data(examples):
            texts = []
            for messages in examples["messages"]:
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to process example: {e}")
                    texts.append("") 
            return {"text": texts}
        
        return dataset.map(
            format_chat_data,
            batched=True,
            batch_size=100,  # バッチサイズを適切に設定
            num_proc=self.num_proc,  # 並列処理を追加
            remove_columns=column_names,
            desc="Processing messages"  # 進捗表示
        )
    
    def _process_qa_format(self, dataset, column_names):
        """question/answerカラムの処理"""
        logger.info("Processing Q&A format dataset (question/answer columns)")
        
        def format_qa_data(examples):
            texts = []
            for question, answer in zip(examples["question"], examples["answer"]):
                try:
                    messages = [
                        {"role": "user", "content": str(question)},
                        {"role": "assistant", "content": str(answer)}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to process Q&A example: {e}")
                    texts.append("")
            return {"text": texts}
        
        return dataset.map(
            format_qa_data,
            batched=True,
            batch_size=100,
            num_proc=self.num_proc,  # 並列処理を追加
            remove_columns=column_names,
            desc="Processing Q&A"
        )
    
    def _process_io_format(self, dataset, column_names):
        """input/outputカラムの処理"""
        logger.info("Processing input/output format dataset")
        
        def format_io_data(examples):
            texts = []
            for input_text, output_text in zip(examples["input"], examples["output"]):
                try:
                    messages = [
                        {"role": "user", "content": str(input_text)},
                        {"role": "assistant", "content": str(output_text)}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to process input/output example: {e}")
                    texts.append("")
            return {"text": texts}
        
        return dataset.map(
            format_io_data,
            batched=True,
            batch_size=100,
            num_proc=self.num_proc,  # 並列処理を追加
            remove_columns=column_names,
            desc="Processing input/output"
        )
    
    def _process_text_format(self, dataset, column_names):
        """textカラムの処理（事前処理済み）"""
        logger.info("Dataset already has 'text' column, using as-is")
        columns_to_remove = [col for col in column_names if col != "text"]
        if columns_to_remove:
            return dataset.remove_columns(columns_to_remove)
        return dataset
    
    def _process_pc_format(self, dataset, column_names):
        """prompt/completionカラムの処理（OpenAI形式）"""
        logger.info("Processing prompt/completion format dataset")
        
        def format_pc_data(examples):
            texts = []
            for prompt, completion in zip(examples["prompt"], examples["completion"]):
                try:
                    messages = [
                        {"role": "user", "content": str(prompt)},
                        {"role": "assistant", "content": str(completion)}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to process prompt/completion example: {e}")
                    texts.append("")
            return {"text": texts}
        
        return dataset.map(
            format_pc_data,
            batched=True,
            batch_size=100,
            num_proc=self.num_proc,  # 並列処理を追加
            remove_columns=column_names,
            desc="Processing prompt/completion"
        )
    
    def _process_ir_format(self, dataset, column_names):
        """instruction/responseカラムの処理（Alpaca形式）"""
        logger.info("Processing instruction/response format dataset")
        
        def format_ir_data(examples):
            texts = []
            has_input = "input" in column_names
            
            for i in range(len(examples["instruction"])):
                try:
                    instruction = str(examples["instruction"][i])
                    response = str(examples["response"][i])
                    
                    # inputカラムがある場合は結合
                    if has_input and examples.get("input", [None])[i]:
                        user_content = f"{instruction}\n\n{examples['input'][i]}"
                    else:
                        user_content = instruction
                    
                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": response}
                    ]
                    text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                except Exception as e:
                    logger.warning(f"Failed to process instruction/response example: {e}")
                    texts.append("")
            return {"text": texts}
        
        return dataset.map(
            format_ir_data,
            batched=True,
            batch_size=100,
            num_proc=self.num_proc,  # 並列処理を追加
            remove_columns=column_names,
            desc="Processing instruction/response"
        )
    
    @staticmethod
    def validate_dataset(dataset):
        """処理後のデータセットを検証"""
        if len(dataset) == 0:
            raise ValueError("Processed dataset is empty!")
        
        # サンプルを表示
        logger.info("Sample processed text (first 500 chars):")
        logger.info(dataset[0]["text"][:500])
        
        return dataset