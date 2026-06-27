import json
import ollama
import re
import os
import csv
import sys
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


class LLMSlicer:
    def __init__(self, model_name="qwen3-coder:30b"):
        self.model_name = model_name

    def process_batch_slicing(self, input_csv, context_jsonl, output_csv):
        if not os.path.exists(input_csv) or not os.path.exists(context_jsonl):
            print("[!] Input file or context file does not exist.")
            return

        with open(input_csv, 'r', encoding='utf-8') as f_in, \
                open(context_jsonl, 'r', encoding='utf-8') as f_ctx, \
                open(output_csv, 'w', encoding='utf-8', newline='') as f_out:

            total_samples = sum(1 for _ in csv.DictReader(f_in))
            f_in.seek(0)

            reader = csv.DictReader(f_in)
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            progress_bar = tqdm(zip(reader, f_ctx), total=total_samples, desc="[*] Slicing Progress", unit="row")

            for i, (csv_row, ctx_line) in enumerate(progress_bar):
                source_code = csv_row.get('text', '')
                cpg_context = json.loads(ctx_line)

                sliced_text = self.slice_single_item(source_code, cpg_context)

                csv_row['text'] = sliced_text
                writer.writerow(csv_row)

                if i % 10 == 0:
                    f_out.flush()

        print(f"\n[+] Batch slicing completed. Results saved to {output_csv}")
        self.compare_results(input_csv, output_csv)

    def slice_single_item(self, source_code, cpg_context):
        if len(source_code) > 10000:
            return " ".join(source_code.split())

        dfg_list = cpg_context.get('dfg', [])
        ast_list = cpg_context.get('ast', [])
        is_context_empty = len(ast_list) == 0

        sinks = [dfg['to'] for dfg in dfg_list if 'FUN' in str(dfg.get('to', ''))]
        sink_desc = f"Sink points: {', '.join(set(sinks[-2:]))}" if sinks else "critical terminal operations"

        context_hint = ""
        if is_context_empty:
            context_hint = (
                "\n[Note: Graph analysis failed. Please use expert logic to preserve "
                "the chain from variable inputs to the target Sink functions.]\n"
            )

        system_prompt = f"""You are a C/C++ code distillation engine. {context_hint}
        - TASK: Remove all noise irrelevant to the data flow of the target.
        - TARGET: {sink_desc}
        - RULE 1: ONLY output the simplified code inside [RESULT] and [/RESULT] tags.
        - RULE 2: DO NOT provide any explanation, summary, or natural language.
        - RULE 3: DO NOT use Markdown formatting (no ```, no +/-).
        - RULE 4: Ensure the output code is syntactically logical and complete."""

        user_content = f"### Source Code:\n{source_code}\n\n### Task:\nSimplify code between [RESULT] tags.\nTarget: {sink_desc}"

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content},
            ])
            raw_content = response['message']['content']

            match = re.search(r"\[RESULT\](.*?)\[/RESULT\]", raw_content, re.DOTALL | re.IGNORECASE)
            if match:
                sliced_code = match.group(1).strip()
            else:
                split_keywords = ["精简后代码", "精简后的代码", "Resulting code"]
                temp_code = raw_content
                for kw in split_keywords:
                    if kw in temp_code:
                        temp_code = temp_code.split(kw)[-1].replace("：", "").replace(":", "").strip()
                        break

                code_blocks = re.findall(r"```(?:cpp|c)?\n(.*?)\n```", temp_code, re.DOTALL)
                if code_blocks:
                    sliced_code = code_blocks[0].strip()
                else:
                    lines = [l for l in temp_code.split('\n') if not re.match(r'^\s*\[.*\]', l)]
                    sliced_code = '\n'.join(lines).strip()

            sliced_code = re.sub(r"```(cpp|c)?", "", sliced_code)
            clean_lines = []
            for line in sliced_code.split('\n'):
                line = re.sub(r"^[+\-#*]{1,2}(?=\s|\w)", "", line).strip()
                if line:
                    clean_lines.append(line)

            sliced_code = " ".join(clean_lines)
            sliced_code = " ".join(sliced_code.split())

            return sliced_code

        except Exception as e:
            print(f"[!] LLM inference request failed: {e}")
            return " ".join(source_code.split())

    def compare_results(self, input_csv, output_csv):
        total_orig = 0
        total_sliced = 0
        count = 0

        try:
            with open(input_csv, 'r', encoding='utf-8') as f_in, \
                    open(output_csv, 'r', encoding='utf-8') as f_out:

                reader_in = csv.DictReader(f_in)
                reader_out = csv.DictReader(f_out)

                for row_in, row_out in zip(reader_in, reader_out):
                    orig_len = len(row_in['text'])
                    sliced_len = len(row_out['text'])

                    total_orig += orig_len
                    total_sliced += sliced_len
                    count += 1

            if count > 0:
                avg_reduction = ((total_orig - total_sliced) / total_orig * 100) if total_orig > 0 else 0

                print("\n" + "=" * 60)
                print(f"  Slicing Tasks Summary (Total Rows: {count})")
                print("-" * 60)
                print(f"  Total Original Characters: {total_orig:,}")
                print(f"  Total Sliced Characters:   {total_sliced:,}")
                print(f"  Overall Reduction Rate:    {avg_reduction:.2f}%")
                print("=" * 60 + "\n")
            else:
                print("\n[!] No matching data available for evaluation.")

        except Exception as e:
            print(f"\n[!] Evaluation failed due to internal error: {e}")


if __name__ == "__main__":
    slicer = LLMSlicer(model_name="qwen3-coder:30b")
    slicer.process_batch_slicing(
        input_csv="../dataset/reveal/reveal_train.csv",
        context_jsonl="reveal_context_train.jsonl",
        output_csv="reveal_slice_train.csv"
    )