from pathlib import Path
import shutil

import pandas as pd

from BLRun.runner import Runner


CELLORACLE_SCRIPT = Path(__file__).resolve().parent.parent / "Algorithms" / "CELLORACLE" / "runCellOracle.py"


class CellOracleRunner(Runner):
    """Concrete runner for CellOracle prior-informed GRN construction."""

    def generateInputs(self):
        expression_file = self.working_dir / "ExpressionData.csv"
        if expression_file.exists():
            return

        source_expression = self.input_dir / self.exprData
        if not source_expression.exists():
            raise FileNotFoundError(f"Expression data file not found: {source_expression}")

        shutil.copy2(source_expression, expression_file)

    def run(self):
        max_edges_per_target = self._resolve_max_edges_per_target()
        args = [
            "python /runCellOracle.py",
            "--inFile=/usr/working_dir/ExpressionData.csv",
            "--outFile=/usr/working_dir/outFile.tsv",
            "--detailsFile=/usr/working_dir/celloracle_links.tsv",
            f"--species={self.params.get('species', 'human')}",
            f"--baseGrn={self.params.get('baseGrn', 'auto')}",
            f"--clusterName={self.params.get('clusterName', 'Global')}",
            f"--maxGenes={int(self.params.get('maxGenes', 3000))}",
            f"--maxCells={int(self.params.get('maxCells', 30000))}",
            f"--minCells={int(self.params.get('minClusterCells', 50))}",
            f"--pValueCutoff={float(self.params.get('pValueCutoff', 0.05))}",
        ]
        if max_edges_per_target is not None:
            args.append(f"--maxRegulatorsPerTarget={int(max_edges_per_target)}")

        cmd_to_run = " ".join([
            "docker run --rm",
            f"-v {self.working_dir}:/usr/working_dir",
            f"-v {CELLORACLE_SCRIPT}:/runCellOracle.py:ro",
            f'{self.image} /bin/sh -c "time -v -o',
            "/usr/working_dir/time.txt",
            *args,
            '"',
        ])

        self._run_docker(cmd_to_run)

    def parseOutput(self):
        out_file = self.working_dir / "outFile.tsv"
        if not out_file.exists():
            print(str(out_file) + " does not exist, skipping...")
            return

        edges = pd.read_csv(out_file, sep="\t")
        self._write_ranked_edges(edges)
