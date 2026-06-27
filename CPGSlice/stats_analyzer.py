import os
import csv
import sys

# Set CSV field limit to the maximum allowed by the system
csv.field_size_limit(sys.maxsize)

def analyze_slicing_stats(input_csv, sliced_csv, report_file="output.txt"):
    if not os.path.exists(input_csv) or not os.path.exists(sliced_csv):
        print(f"[!] File not found: {input_csv} or {sliced_csv}")
        return

    total_orig = 0
    total_sliced = 0
    count = 0
    negative_rows = []

    with open(input_csv, 'r', encoding='utf-8') as f_in, \
            open(sliced_csv, 'r', encoding='utf-8') as f_out, \
            open(report_file, 'w', encoding='utf-8') as f_rep:

        reader_in = csv.DictReader(f_in)
        reader_out = csv.DictReader(f_out)

        # Write report header
        f_rep.write("=" * 85 + "\n")
        f_rep.write(f"{'Row':<8} | {'Original (Char)':<18} | {'Sliced (Char)':<18} | {'Reduction %':<15}\n")
        f_rep.write("-" * 85 + "\n")

        print(f"[*] Analyzing data and writing to {report_file}...")

        for i, (row_in, row_out) in enumerate(zip(reader_in, reader_out)):
            orig_text = row_in.get('text', '')
            sliced_text = row_out.get('text', '')

            orig_len = len(orig_text)
            sliced_len = len(sliced_text)

            reduction = ((orig_len - sliced_len) / orig_len * 100) if orig_len > 0 else 0

            f_rep.write(f"{i + 1:<8} | {orig_len:<18} | {sliced_len:<18} | {reduction:>12.2f}%\n")

            total_orig += orig_len
            total_sliced += sliced_len
            count += 1

            # Track anomalies where code size expanded post-slicing
            if reduction < 0:
                negative_rows.append((i + 1, reduction))

        # Write final statistics summary
        if count > 0:
            avg_reduction = ((total_orig - total_sliced) / total_orig * 100) if total_orig > 0 else 0
            f_rep.write("-" * 85 + "\n")
            f_rep.write(f"TOTAL    | {total_orig:<18} | {total_sliced:<18} | {avg_reduction:>12.2f}%\n")
            f_rep.write("=" * 85 + "\n")

            if negative_rows:
                f_rep.write(f"\n[!] Found {len(negative_rows)} rows with negative reduction (size increased):\n")
                for row_num, val in negative_rows:
                    f_rep.write(f"Row {row_num}: {val:.2f}%\n")

    print(f"[+] Analysis completed. Statistical report saved to: {report_file}")


if __name__ == "__main__":
    analyze_slicing_stats(
        input_csv="../dataset/reveal/reveal_train.csv",
        sliced_csv="reveal_slice_train.csv",
        report_file="reveal_reduction_rate.txt"
    )