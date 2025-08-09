#!/usr/bin/env python
"""Analyze test coverage and identify gaps."""

import os
from pathlib import Path

def analyze_coverage():
    """Analyze which modules have tests and which don't."""
    
    src_dir = Path("src/brain_go_brrr")
    test_dir = Path("tests")
    
    # Get all source modules
    source_modules = set()
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" not in str(py_file):
            rel_path = py_file.relative_to(src_dir)
            module = str(rel_path).replace(".py", "").replace("/", ".")
            if module != "__init__":
                source_modules.add(module)
    
    # Get all test files
    test_modules = set()
    for py_file in test_dir.rglob("test_*.py"):
        if "__pycache__" not in str(py_file):
            rel_path = py_file.relative_to(test_dir)
            # Extract what's being tested from the name
            test_name = str(rel_path.name).replace("test_", "").replace(".py", "")
            test_modules.add(test_name)
    
    # Categorize modules
    api_modules = {m for m in source_modules if m.startswith("api.")}
    core_modules = {m for m in source_modules if m.startswith("core.")}
    models_modules = {m for m in source_modules if m.startswith("models.")}
    services_modules = {m for m in source_modules if m.startswith("services.")}
    data_modules = {m for m in source_modules if m.startswith("data.")}
    utils_modules = {m for m in source_modules if m.startswith("utils.")}
    cli_modules = {m for m in source_modules if m == "cli" or m.startswith("cli.")}
    
    print("=" * 60)
    print("TEST COVERAGE ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal source modules: {len(source_modules)}")
    print(f"Total test files: {len(list(test_dir.rglob('test_*.py')))}")
    
    print("\n📊 MODULE BREAKDOWN:")
    print(f"  API:      {len(api_modules)} modules")
    print(f"  Core:     {len(core_modules)} modules")
    print(f"  Models:   {len(models_modules)} modules")
    print(f"  Services: {len(services_modules)} modules")
    print(f"  Data:     {len(data_modules)} modules")
    print(f"  Utils:    {len(utils_modules)} modules")
    print(f"  CLI:      {len(cli_modules)} modules")
    
    # Find modules without tests
    print("\n❌ MODULES WITHOUT DIRECT TESTS:")
    
    # Check each category
    for category, modules in [
        ("Core", core_modules),
        ("Models", models_modules),
        ("Services", services_modules),
        ("Data", data_modules),
        ("Utils", utils_modules),
    ]:
        missing = []
        for module in sorted(modules):
            # Check if there's a corresponding test file
            module_name = module.split(".")[-1]
            test_path = test_dir / "unit" / f"test_{module_name}.py"
            if not test_path.exists():
                # Also check integration tests
                test_path = test_dir / "integration" / f"test_{module_name}.py"
                if not test_path.exists():
                    missing.append(module)
        
        if missing:
            print(f"\n{category}:")
            for m in missing[:10]:  # Show first 10
                print(f"  - {m}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
    
    # Prioritize what to test
    print("\n🎯 PRIORITY MODULES TO TEST (critical path):")
    priority = [
        "core.config",
        "core.edf_loader",
        "core.exceptions",
        "models.eegpt_model",
        "models.linear_probe",
        "services.yasa_adapter",
        "data.tuab_dataset",
        "data.cache_manager",
        "utils.preprocessing",
    ]
    
    for module in priority:
        if module in source_modules:
            test_exists = (test_dir / "unit" / f"test_{module.split('.')[-1]}.py").exists()
            status = "✅" if test_exists else "❌"
            print(f"  {status} {module}")

if __name__ == "__main__":
    analyze_coverage()