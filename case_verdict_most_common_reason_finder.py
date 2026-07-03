# json_summary_agent.py

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class JSONSummaryAgent:
    def __init__(
        self,
        input_path: str,
        output_excel: str = "case_summary.xlsx",
        similarity_threshold: float = 0.95
    ):
        self.input_path = Path(input_path)
        self.output_excel = output_excel
        self.similarity_threshold = similarity_threshold
        self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    def load_json_files(self):
        if self.input_path.is_file() and self.input_path.suffix.lower() == ".json":
            return [self.input_path]

        if self.input_path.is_dir():
            return list(self.input_path.glob("*.json"))

        raise FileNotFoundError(f"Invalid input path: {self.input_path}")

    def find_first_key(self, data, target_key):
        if isinstance(data, dict):
            for key, value in data.items():
                if key == target_key:
                    return value

                result = self.find_first_key(value, target_key)
                if result is not None:
                    return result

        elif isinstance(data, list):
            for item in data:
                result = self.find_first_key(item, target_key)
                if result is not None:
                    return result

        return None

    def normalize_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"[^a-z0-9₹.,\s/-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def extract_records(self):
        records = []

        for file_path in self.load_json_files():
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                case_verdict = self.find_first_key(data, "case_verdict")

                if str(case_verdict).strip() != "0":
                    continue

                records.append({
                    "AL numbers": self.find_first_key(data, "case_number") or "",
                    "reason": self.find_first_key(data, "summary") or "",
                    "Case Verdict": case_verdict,
                    "amounts_match": self.find_first_key(data, "amounts_match"),
                    "net_corrected": self.find_first_key(data, "net_corrected"),
                    "total_line_items": self.find_first_key(data, "total_line_items"),
                    "overall_confidence": self.find_first_key(data, "overall_confidence"),
                })

            except Exception as e:
                print(f"Skipped file: {file_path} | Error: {e}")

        return pd.DataFrame(records)

    def extract_flagged_line_item_count(self, reason):
        match = re.search(r"(\d+)\s+line item\(s\)\s+are flagged", reason.lower())
        return int(match.group(1)) if match else 0

    def extract_breakdown_mismatch_count(self, reason):
        match = re.search(r"(\d+)\s+itemised breakdown\(s\)\s+do not add up", reason.lower())
        return int(match.group(1)) if match else 0

    def get_specific_category(self, row):
        reason = self.normalize_text(row.get("reason", ""))

        amounts_match = row.get("amounts_match")
        net_corrected = row.get("net_corrected")
        total_line_items = row.get("total_line_items")
        confidence = row.get("overall_confidence")

        flagged_count = self.extract_flagged_line_item_count(reason)
        breakdown_count = self.extract_breakdown_mismatch_count(reason)

        if total_line_items == 0 or "0 line item" in reason:
            return "Zero Line Items Extracted - Printed Total Not Reconciled"

        if "printed net" in reason and "inconsistent" in reason and "recomputed" in reason:
            return "Printed Net Payable Recomputed From Itemised Total"

        if "do not reconcile" in reason or "do not reconcile with the printed total" in reason:
            return "Line Item Total Does Not Match Printed Total"

        if breakdown_count > 0:
            return "Itemised Breakdown Header Total Mismatch"

        if flagged_count > 0 and amounts_match is True:
            return "Reconciled Bill With Flagged Line Items"

        if amounts_match is True and confidence is not None and float(confidence) < 0.90:
            return "Reconciled Bill With Low Extraction Confidence"

        if amounts_match is True:
            return "Reconciled Bill Pending Reviewer Attention"

        return "Specific Billing Review Required"

    def generate_tfidf_specific_label(self, reasons):
        clean_reasons = [self.normalize_text(r) for r in reasons if str(r).strip()]

        if not clean_reasons:
            return "Specific Billing Review Required"

        stop_words = [
            "case", "flagged", "needs", "reviewer", "attention",
            "consolidated", "bill", "line", "item", "items",
            "totalling", "gross", "net", "payable", "billing",
            "type", "assessed", "extraction", "confidence"
        ]

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(2, 5),
                max_features=30
            )

            tfidf = vectorizer.fit_transform(clean_reasons)
            scores = np.asarray(tfidf.sum(axis=0)).flatten()
            terms = vectorizer.get_feature_names_out()

            ranked_terms = [
                terms[i]
                for i in scores.argsort()[::-1]
                if not any(sw in terms[i].split() for sw in stop_words)
            ]

            if ranked_terms:
                return ranked_terms[0].title()

        except Exception:
            pass

        return "Specific Billing Review Required"

    def merge_unknown_categories_semantically(self, df):
        unknown_mask = df["reason_category"] == "Specific Billing Review Required"

        if unknown_mask.sum() <= 1:
            return df

        unknown_reasons = df.loc[unknown_mask, "reason"].fillna("").astype(str).tolist()
        unknown_indexes = df.loc[unknown_mask].index.tolist()

        embeddings = self.embedding_model.encode(
            unknown_reasons,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=True
        )

        sim_matrix = cosine_similarity(embeddings)

        visited = set()
        cluster_id = 1

        for i in range(len(unknown_reasons)):
            if i in visited:
                continue

            stack = [i]
            visited.add(i)
            cluster_members = []

            while stack:
                current = stack.pop()
                cluster_members.append(current)

                for j in range(len(unknown_reasons)):
                    if j not in visited and sim_matrix[current][j] >= self.similarity_threshold:
                        visited.add(j)
                        stack.append(j)

            cluster_texts = [unknown_reasons[x] for x in cluster_members]
            label = self.generate_tfidf_specific_label(cluster_texts)

            if label == "Specific Billing Review Required":
                label = f"Specific Billing Review Required - Cluster {cluster_id}"

            for member in cluster_members:
                df.at[unknown_indexes[member], "reason_category"] = label

            cluster_id += 1

        return df

    def create_reason_categories(self, df):
        if df.empty:
            df["reason_category"] = ""
            df["reason_category_count"] = 0
            df["reason_category_ranking"] = ""
            return df

        df["reason_category"] = df.apply(self.get_specific_category, axis=1)

        df = self.merge_unknown_categories_semantically(df)

        category_counts = Counter(df["reason_category"])

        category_ranking = {
            category: rank
            for rank, (category, count) in enumerate(
                category_counts.most_common(),
                start=1
            )
        }

        df["reason_category_count"] = df["reason_category"].map(category_counts)
        df["reason_category_ranking"] = df["reason_category"].map(category_ranking)

        return df

    def write_excel(self, df):
        final_columns = [
            "AL numbers",
            "reason",
            "Case Verdict",
            "reason_category",
            "reason_category_count",
            "reason_category_ranking"
        ]

        df = df[final_columns]

        with pd.ExcelWriter(self.output_excel, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Summary_Output")

            worksheet = writer.sheets["Summary_Output"]

            for column_cells in worksheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))

                worksheet.column_dimensions[column_letter].width = min(max_length + 3, 90)

        print(f"Excel generated successfully: {self.output_excel}")

    def run(self):
        df = self.extract_records()

        if df.empty:
            raise ValueError("No JSON files found with case_verdict = 0")

        df = self.create_reason_categories(df)
        self.write_excel(df)


if __name__ == "__main__":
    agent = JSONSummaryAgent(
        input_path="input_jsons",
        output_excel="case_summary.xlsx",
        similarity_threshold=0.95
    )

    agent.run()