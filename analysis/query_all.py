import argparse

import duckdb
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Query a DuckDB database and display all rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db",       default="west.db", help="Path to the DuckDB database")
    p.add_argument("--table",    default="segment_data",           help="Table to query")
    p.add_argument("--dataset",  default=None,                     help="Filter by dataset name (default: all)")
    p.add_argument("--order-by", default="iteration, segment",     help="ORDER BY clause")
    p.add_argument("--output",   default=None,                     help="Save results to CSV at this path (default: print to stdout)")
    return p


def fetch_all(
    con: duckdb.DuckDBPyConnection,
    table: str,
    dataset: str | None,
    order_by: str,
) -> pd.DataFrame:
    where = f"WHERE dataset = '{dataset}'" if dataset else ""
    return con.execute(f"""
        SELECT iteration, segment, dataset, values_json
        FROM {table}
        {where}
        ORDER BY {order_by}
    """).fetchdf()


def main() -> None:
    args = build_parser().parse_args()

    with duckdb.connect(args.db) as con:
        print("Tables:", con.execute("SHOW TABLES").fetchall())

        df = fetch_all(con, args.table, args.dataset, args.order_by)

    print(f"Rows returned: {len(df)}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df)


if __name__ == "__main__":
    main()
