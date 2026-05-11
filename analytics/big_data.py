"""Optional big-data style preprocessing for large ICU CSV files."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pandas as pd


def chunked_csv_summary(csv_path: str | Path, chunksize: int = 25_000) -> pd.DataFrame:
    summaries = []
    for chunk_id, chunk in enumerate(
        pd.read_csv(csv_path, chunksize=chunksize), start=1
    ):
        numeric = chunk.select_dtypes(include="number")
        summaries.append(
            {
                "chunk_id": chunk_id,
                "rows": len(chunk),
                "numeric_columns": len(numeric.columns),
                "missing_values": int(chunk.isna().sum().sum()),
            }
        )
    return pd.DataFrame(summaries)


def pyspark_preprocess_simulation(
    input_path: str | Path,
    output_path: str | Path = "outputs/pyspark_batch_summary.csv",
) -> Path:
    """Use PySpark when installed; otherwise fall back to chunked Pandas analytics."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("pyspark") is not None:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("ICU-TwinAI-Batch").getOrCreate()
        df = spark.read.csv(str(input_path), header=True, inferSchema=True)
        summary = pd.DataFrame(
            [{"rows": df.count(), "columns": len(df.columns), "engine": "pyspark"}]
        )
        spark.stop()
    else:
        summary = chunked_csv_summary(input_path)
        summary["engine"] = "pandas_chunked_fallback"
    summary.to_csv(output, index=False)
    return output
