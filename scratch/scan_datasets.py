import os

def scan_mock_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mock_dir = os.path.join(base_dir, "MockData")
    
    datasets = []
    categories = ["electronics", "fmcg"]
    for cat in categories:
        cat_path = os.path.join(mock_dir, cat)
        if not os.path.exists(cat_path):
            continue
        for prod in os.listdir(cat_path):
            if prod.startswith('.') or not os.path.isdir(os.path.join(cat_path, prod)):
                continue
            prod_path = os.path.join(cat_path, prod)
            for scenario in os.listdir(prod_path):
                if scenario.startswith('.') or not os.path.isdir(os.path.join(prod_path, scenario)):
                    continue
                datasets.append((cat, prod, scenario))
                
    print(f"Total datasets found: {len(datasets)}")
    for d in sorted(datasets):
        print(f" - {d[0]} / {d[1]} / {d[2]}")

if __name__ == '__main__':
    scan_mock_data()
