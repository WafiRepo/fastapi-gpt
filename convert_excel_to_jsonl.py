#!/usr/bin/env python3
"""
Script untuk mengkonversi Excel file ke JSONL format untuk evaluation.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas openpyxl")
    sys.exit(1)


def parse_reference(ref_value: Any) -> List[str]:
    """
    Parse reference value yang mungkin berupa string JSON atau list.
    """
    if pd.isna(ref_value):
        return []
    
    if isinstance(ref_value, list):
        return ref_value
    
    if isinstance(ref_value, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(ref_value)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and "reference" in parsed:
                return parsed["reference"] if isinstance(parsed["reference"], list) else []
            else:
                return [str(parsed)]
        except json.JSONDecodeError:
            # If not JSON, treat as single string
            return [ref_value] if ref_value.strip() else []
    
    return []


def convert_excel_to_jsonl(excel_path: str, output_path: str) -> None:
    """
    Convert Excel file ke JSONL format untuk evaluation.
    """
    print(f"Reading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    
    print(f"Found {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Required columns
    required_cols = ["id", "stage", "inquiry"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert to JSONL
    items: List[Dict[str, Any]] = []
    
    for idx, row in df.iterrows():
        item: Dict[str, Any] = {
            "id": str(row.get("id", f"item_{idx+1}")),
            "stage": str(row.get("stage", "")),
            "inquiry": str(row.get("inquiry", "")),
        }
        
        # Optional fields
        if "lang" in df.columns:
            item["lang"] = str(row.get("lang", "en"))
        
        if "prompt_type" in df.columns:
            item["prompt_type"] = str(row.get("prompt_type", ""))
        
        if "context" in df.columns:
            item["context"] = str(row.get("context", ""))
        
        # Parse reference
        if "reference" in df.columns:
            reference = parse_reference(row.get("reference"))
            if reference:
                item["reference"] = reference
        
        items.append(item)
    
    # Write to JSONL
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
    
    print(f"\nConverted {len(items)} items to JSONL")
    print(f"Output saved to: {output_path}")
    
    # Show sample
    if items:
        print("\nSample item (first row):")
        print(json.dumps(items[0], indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Excel file to JSONL format for evaluation"
    )
    parser.add_argument("--input", required=True, help="Input Excel file path")
    parser.add_argument("--output", help="Output JSONL file path (default: input filename with .jsonl extension)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1
    
    if args.output:
        output_path = args.output
    else:
        output_path = args.input.replace(".xlsx", ".jsonl").replace(".xls", ".jsonl")
    
    try:
        convert_excel_to_jsonl(args.input, output_path)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
