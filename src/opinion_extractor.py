import re
from typing import Literal

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


class OpinionExtractor:

    # SET THE FOLLOWING CLASS VARIABLE to "FT" if you implemented a fine-tuning approach
    method: Literal["NOFT", "FT"] = "FT"

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def __init__(self, cfg) -> None:
        self.method = "FT"
        self.cfg = cfg
        self.base_model_id = "Qwen/Qwen3-1.7B"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.aspects = ["Price", "Food", "Service"]
        self.labels = ["Positive", "Negative", "Mixed", "No Opinion"]
        self.model = None

    def _build_prompt(self, review: str) -> str:
        return (
            "You are an aspect-based opinion extraction system.\n"
            "Given a French restaurant review, classify the overall opinion\n"
            "for each of the 3 aspects: Price, Food, Service.\n"
            "Each aspect must be classified as exactly one of:\n"
            "Positive, Negative, Mixed, No Opinion.\n\n"
            "Definitions:\n"
            "- Positive: the review contains positive opinion(s) on this aspect and no negative ones.\n"
            "- Negative: the review contains negative opinion(s) on this aspect and no positive ones.\n"
            "- Mixed: the review contains both positive and negative opinions on this aspect.\n"
            "- No Opinion: the review does not express any opinion on this aspect.\n\n"
            f"Review: {review}\n\n"
            "Answer in exactly this format and nothing else:\n"
            "Price: <label>\n"
            "Food: <label>\n"
            "Service: <label>\n"
        )

    def _find_column(self, columns: list[str], expected_name: str) -> str:
        for col in columns:
            if col.strip().lower() == expected_name.lower():
                return col
        raise ValueError(f"Missing required column: {expected_name}")

    def _to_dataframe(self, data) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            if isinstance(data[0], dict):
                return pd.DataFrame(data)
            if isinstance(data[0], str):
                return pd.DataFrame({"Review": data})
        raise TypeError("Input data must be a pandas DataFrame, list of dicts, or list of strings.")

    def _normalize_label(self, raw_label: str) -> str:
        if raw_label is None:
            return "No Opinion"
        cleaned = raw_label.strip().splitlines()[0].strip()
        cleaned = cleaned.strip(" .,:;!?")
        mapping = {
            "positive": "Positive",
            "negative": "Negative",
            "mixed": "Mixed",
            "no opinion": "No Opinion",
            "no-opinion": "No Opinion",
            "no_opinion": "No Opinion",
        }
        return mapping.get(cleaned.lower(), "No Opinion")

    def _parse_prediction(self, generated: str) -> dict:
        parsed = {}
        for aspect in self.aspects:
            match = re.search(
                rf"^{re.escape(aspect)}:\s*(.+)",
                generated,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            label = self._normalize_label(match.group(1) if match else "")
            if label not in self.labels:
                label = "No Opinion"
            parsed[aspect] = label
        return parsed

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Trains the model, if OpinionExtractor.method=="FT"
        """
        train_df = self._to_dataframe(train_data)
        dev_df = self._to_dataframe(val_data)

        review_col_train = self._find_column(list(train_df.columns), "Review")
        review_col_dev = self._find_column(list(dev_df.columns), "Review")
        train_aspect_cols = {
            aspect: self._find_column(list(train_df.columns), aspect) for aspect in self.aspects
        }
        dev_aspect_cols = {
            aspect: self._find_column(list(dev_df.columns), aspect) for aspect in self.aspects
        }

        train_records = []
        for _, row in train_df.iterrows():
            prompt = self._build_prompt(str(row[review_col_train]))
            completion = (
                f"Price: {row[train_aspect_cols['Price']]}\n"
                f"Food: {row[train_aspect_cols['Food']]}\n"
                f"Service: {row[train_aspect_cols['Service']]}\n"
            )
            train_records.append({"text": prompt + completion})

        dev_records = []
        for _, row in dev_df.iterrows():
            prompt = self._build_prompt(str(row[review_col_dev]))
            completion = (
                f"Price: {row[dev_aspect_cols['Price']]}\n"
                f"Food: {row[dev_aspect_cols['Food']]}\n"
                f"Service: {row[dev_aspect_cols['Service']]}\n"
            )
            dev_records.append({"text": prompt + completion})

        train_dataset = Dataset.from_list(train_records)
        dev_dataset = Dataset.from_list(dev_records)

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map=None,
        )

        num_devices = max(1, torch.cuda.device_count())
        base_per_device_batch = 2
        desired_effective_batch = 16
        gradient_accumulation_steps = max(
            1, desired_effective_batch // (base_per_device_batch * num_devices)
        )
        effective_batch = base_per_device_batch * num_devices * gradient_accumulation_steps
        learning_rate = 2e-4 * (effective_batch / 16)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir="./lora_checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=base_per_device_batch,
            per_device_eval_batch_size=base_per_device_batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            logging_steps=10,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=512,
        )
        trainer.train()

        model.save_pretrained("./lora_adapter")
        self.tokenizer.save_pretrained("./lora_adapter")
        self.model = model
        self.model.eval()

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def predict(self, texts: list[str]) -> list[dict]:
        """
        :param texts: list of reviews from which to extract the opinion values
        :return: a list of dicts, one per input review, containing the opinion values for the 3 aspects.
        """
        input_was_dataframe = isinstance(texts, pd.DataFrame)
        input_index = None

        if input_was_dataframe:
            data_df = texts.copy()
            input_index = data_df.index
            review_col = self._find_column(list(data_df.columns), "Review")
            review_texts = data_df[review_col].astype(str).tolist()
        elif isinstance(texts, list) and (not texts or isinstance(texts[0], str)):
            review_texts = [str(t) for t in texts]
        elif isinstance(texts, list) and texts and isinstance(texts[0], dict):
            data_df = pd.DataFrame(texts)
            review_col = self._find_column(list(data_df.columns), "Review")
            review_texts = data_df[review_col].astype(str).tolist()
        else:
            raise TypeError("predict expects a list[str], list[dict], or a pandas DataFrame.")

        if self.model is None:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16,
                device_map=None,
            )
            self.model = PeftModel.from_pretrained(base, "./lora_adapter")
            self.model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        predictions = []
        for review in review_texts:
            prompt = self._build_prompt(review)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=480,
            ).to(device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            predictions.append(self._parse_prediction(generated))

        if input_was_dataframe:
            return pd.DataFrame(predictions, columns=self.aspects, index=input_index)
        return predictions
