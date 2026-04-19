import csv
from collections import defaultdict

import app


TEST_FILE = "query_test_cases.csv"


def to_bool(text):
    return str(text).strip().lower() == "true"


def safe_int(value):
    try:
        if str(value).strip() == "":
            return None
        return int(float(value))
    except Exception:
        return None


def evaluate():
    with open(TEST_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    parse_totals = defaultdict(int)
    parse_hits = defaultdict(int)

    endpoint_total = 0
    endpoint_hit = 0

    failed_checks = []

    client = app.app.test_client()

    for idx, row in enumerate(rows, start=2):
        query = (row.get("query") or "").strip()
        if not query:
            continue

        parsed = app.parse_natural_language_query(query)
        strict_budget = app._is_strict_budget_query(query)
        is_related = app.is_car_related_query(query)

        checks = []

        exp_budget = safe_int(row.get("expected_budget_rupees", ""))
        if exp_budget is not None:
            checks.append(("budget", exp_budget, parsed.get("budget")))

        exp_family = safe_int(row.get("expected_family_size", ""))
        if exp_family is not None:
            checks.append(("family_size", exp_family, parsed.get("family_size")))

        exp_pref = (row.get("expected_preference") or "").strip()
        if exp_pref:
            checks.append(("preference", exp_pref, parsed.get("preference")))

        exp_strict = (row.get("expected_strict_budget") or "").strip().lower()
        if exp_strict in {"true", "false"}:
            checks.append(("strict_budget", to_bool(exp_strict), strict_budget))

        exp_related = (row.get("expected_is_car_related") or "").strip().lower()
        expected_related_bool = None
        if exp_related in {"true", "false"}:
            expected_related_bool = to_bool(exp_related)
            checks.append(("is_car_related", expected_related_bool, is_related))

        row_pass = True
        row_fail_details = []

        for name, expected, actual in checks:
            parse_totals[name] += 1
            if expected == actual:
                parse_hits[name] += 1
            else:
                row_pass = False
                row_fail_details.append(
                    {
                        "check": name,
                        "expected": expected,
                        "actual": actual,
                    }
                )

        if expected_related_bool is not None:
            endpoint_total += 1
            resp = client.post("/recommend", json={"query": query})
            payload = resp.get_json(silent=True) or {}

            if expected_related_bool is False:
                endpoint_ok = (resp.status_code == 400 and payload.get("irrelevant_query") is True)
            else:
                endpoint_ok = (payload.get("irrelevant_query") is not True)

            if endpoint_ok:
                endpoint_hit += 1
            else:
                row_pass = False
                row_fail_details.append(
                    {
                        "check": "endpoint_related_behavior",
                        "expected": "irrelevant" if expected_related_bool is False else "not_irrelevant",
                        "actual": {
                            "status": resp.status_code,
                            "irrelevant_query": payload.get("irrelevant_query"),
                            "error": payload.get("error"),
                        },
                    }
                )

        if not row_pass:
            failed_checks.append(
                {
                    "line": idx,
                    "category": row.get("category", ""),
                    "query": query,
                    "fails": row_fail_details,
                }
            )

    print("=== TEST SUMMARY ===")
    total_checks = sum(parse_totals.values())
    total_hits = sum(parse_hits.values())
    overall_pct = (total_hits / total_checks * 100) if total_checks else 0.0
    print(f"Parser checks: {total_hits}/{total_checks} ({overall_pct:.2f}%)")

    for key in ["budget", "family_size", "preference", "strict_budget", "is_car_related"]:
        t = parse_totals.get(key, 0)
        h = parse_hits.get(key, 0)
        pct = (h / t * 100) if t else 0.0
        print(f"- {key}: {h}/{t} ({pct:.2f}%)")

    endpoint_pct = (endpoint_hit / endpoint_total * 100) if endpoint_total else 0.0
    print(f"Endpoint related/irrelevant behavior: {endpoint_hit}/{endpoint_total} ({endpoint_pct:.2f}%)")

    print("\n=== FAILURES (first 25) ===")
    for item in failed_checks[:25]:
        print(f"Line {item['line']} | {item['category']} | {item['query']}")
        for fail in item["fails"]:
            print(f"  * {fail['check']}: expected={fail['expected']} actual={fail['actual']}")

    print(f"\nTotal failed rows: {len(failed_checks)}")


if __name__ == "__main__":
    evaluate()
