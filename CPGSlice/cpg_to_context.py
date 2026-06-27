import os
import subprocess
import json
import tempfile
import re
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Set CSV field limit to the maximum allowed by the system
csv.field_size_limit(sys.maxsize)

class JoernSlicingExtractor:
    def __init__(self, joern_path="joern", joern_parse_path="joern-parse"):
        self.joern_bin = joern_path
        self.joern_parse = joern_parse_path

    def _run_command(self, cmd):
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return result.stdout

    def _format_code(self, text):
        if not text: return ""
        text = text.replace(';', ';\n').replace('{', '{\n').replace('}', '\n}\n')
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)

    def extract_single_context(self, code_text):
        context = {"ast": [], "cfg": [], "dfg": []}
        formatted_code = self._format_code(code_text)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_file = os.path.join(tmpdir, "temp.cpp")
            with open(source_file, "w", encoding='utf-8') as f:
                f.write(formatted_code)

            cpg_bin = os.path.join(tmpdir, "cpg.bin")
            try:
                parse_proc = subprocess.run([self.joern_parse, source_file, "--output", cpg_bin],
                                            capture_output=True)
                if parse_proc.returncode != 0 or not os.path.exists(cpg_bin):
                    return context

                # Scala script template for Joern CPG property extraction
                scala_template = r'''
import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.nodes._

@main def main() = {
    importCpg("CPG_PATH")
    println("---DATA_START---")
    cpg.method.internal.filter(m => !m.name.contains("<")).foreach { m =>
        m.ast.isExpression.lineNumber.toSet.toList.sorted.foreach { (line: Int) =>
            m.ast.isExpression.lineNumber(line).code.l.headOption.foreach { code =>
                val c = code.trim
                if ((c.contains("=") || c.contains("(")) && c.length > 5) {
                    println(s"AST###L$line###$c")
                }
            }
        }
        m.ast.isControlStructure.foreach { ctrl =>
            val sL = ctrl.lineNumber.getOrElse(-1)
            ctrl.cfgNext.isExpression.headOption.foreach { tgt =>
                val tL = tgt.lineNumber.getOrElse(-1)
                if (sL > 0 && tL > 0 && sL != tL) {
                    println(s"CFG###L$sL###${ctrl.code.take(30)}###L$tL")
                }
            }
        }
        m.ast.isExpression.foreach { expr =>
            expr._reachingDefOut.filter(_.isInstanceOf[CfgNode]).foreach { node =>
                val tgt = node.asInstanceOf[CfgNode]
                val sL = expr.lineNumber.getOrElse(-1)
                val tL = tgt.propertyOption[Any]("LINE_NUMBER").getOrElse("-1").toString
                val sCode = expr.code
                val tCode = tgt.propertyOption[Any]("CODE").getOrElse("").toString
                val isNoise = sCode.contains("<") || sCode.contains("RET") || tCode.contains("RET") || 
                              sCode.matches("^[0-9]+$") || sCode.length < 2
                if (sL != -1 && tL != "-1" && sL.toString != tL && !isNoise) {
                    println(s"DFG###L$sL###$sCode###L$tL")
                }
            }
        }
    }
    println("---DATA_END---")
}
'''
                safe_cpg_path = cpg_bin.replace("\\", "/")
                script_content = scala_template.replace("CPG_PATH", safe_cpg_path)
                script_path = os.path.join(tmpdir, "extract.sc")
                with open(script_path, "w", encoding='utf-8') as f:
                    f.write(script_content)

                raw_output = self._run_command([self.joern_bin, "--script", script_path])
                data_match = re.search(r"---DATA_START---\n(.*?)\n---DATA_END---", raw_output, re.DOTALL)
                if data_match:
                    lines = data_match.group(1).strip().splitlines()
                    ast_seen, cfg_seen, dfg_seen = set(), set(), set()
                    for line in lines:
                        parts = line.split("###")
                        if len(parts) < 3: continue
                        tag = parts[0]
                        if tag == "AST":
                            if parts[1] not in ast_seen:
                                context["ast"].append({"line": parts[1], "code": parts[2]})
                                ast_seen.add(parts[1])
                        elif tag == "CFG":
                            if (parts[1], parts[3]) not in cfg_seen:
                                context["cfg"].append({"from": parts[1], "to": parts[3], "trigger": parts[2]})
                                cfg_seen.add((parts[1], parts[3]))
                        elif tag == "DFG":
                            clean_var = parts[2].replace("&", "").split("[")[0].strip()
                            if len(clean_var) > 1 and (parts[1], clean_var, parts[3]) not in dfg_seen:
                                context["dfg"].append({"from": f"{parts[1]} ({clean_var})", "to": parts[3]})
                                dfg_seen.add((parts[1], clean_var, parts[3]))
                if len(context["ast"]) < 3 or (not context["cfg"] and not context["dfg"]):
                    context = {"ast": [], "cfg": [], "dfg": []}
            except Exception:
                pass
        return context

    def process_batch_csv(self, input_csv, output_json, max_workers=8):
        if not os.path.exists(input_csv):
            print(f"[!] Input file {input_csv} not found.")
            return

        tasks = []
        with open(input_csv, mode='r', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            for row in reader:
                tasks.append(row.get('text', ''))

        total = len(tasks)
        print(f"[*] Total tasks: {total}. Using {max_workers} workers.")

        start_time = time.time()

        with open(output_json, 'w', encoding='utf-8') as f_out:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(self.extract_single_context, tasks, chunksize=10)
                for i, context in enumerate(results):
                    if (i + 1) % 100 == 0:
                        elapsed = time.time() - start_time
                        mins, secs = divmod(int(elapsed), 60)
                        speed = (i + 1) / elapsed
                        print(f"[*] Completed {i + 1}/{total} | Time: {mins}m {secs}s | Speed: {speed:.2f} items/s")

                    f_out.write(json.dumps(context, ensure_ascii=False) + "\n")
                    if (i + 1) % 50 == 0:
                        f_out.flush()

        total_time = time.time() - start_time
        t_mins, t_secs = divmod(int(total_time), 60)
        print(f"[+] Total time: {t_mins}m {t_secs}s | Avg speed: {total / total_time:.2f} items/s")
        print(f"[+] Results saved to {output_json}")


if __name__ == "__main__":
    extractor = JoernSlicingExtractor()
    extractor.process_batch_csv(
        "../dataset/reveal/text_val.csv",
        "reveal_context_val.jsonl",
        max_workers=5
    )