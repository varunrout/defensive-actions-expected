"""
Cleanup utility for defensive-action-expected project.

This script removes temporary files, cache directories, and optionally
data artifacts to reset the project to a clean state.
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def find_pycache_dirs(root: Path) -> List[Path]:
    """Find all __pycache__ directories."""
    return list(root.rglob("__pycache__"))


def find_pytest_cache_dirs(root: Path) -> List[Path]:
    """Find all .pytest_cache directories (excluding .venv)."""
    dirs = []
    for path in root.rglob(".pytest_cache"):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        dirs.append(path)
    return dirs


def find_mypy_cache_dirs(root: Path) -> List[Path]:
    """Find all .mypy_cache directories (excluding .venv)."""
    dirs = []
    for path in root.rglob(".mypy_cache"):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        dirs.append(path)
    return dirs


def find_ipynb_checkpoints(root: Path) -> List[Path]:
    """Find all .ipynb_checkpoints directories (excluding .venv)."""
    dirs = []
    for path in root.rglob(".ipynb_checkpoints"):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        dirs.append(path)
    return dirs


def find_pyc_files(root: Path) -> List[Path]:
    """Find all .pyc files (excluding .venv)."""
    pyc_files = []
    for ext in ["*.pyc", "*.pyo", "*.pyd"]:
        for path in root.rglob(ext):
            if ".venv" in path.parts or ".git" in path.parts:
                continue
            pyc_files.append(path)
    return pyc_files


def find_log_files(root: Path) -> List[Path]:
    """Find all log files (excluding .venv)."""
    log_files = []
    for path in root.rglob("*.log"):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        log_files.append(path)
    return log_files


def find_coverage_files(root: Path) -> List[Path]:
    """Find all coverage files (excluding .venv)."""
    coverage_files = []
    for pattern in [".coverage*", "htmlcov/"]:
        for path in root.rglob(pattern):
            if ".venv" in path.parts or ".git" in path.parts:
                continue
            coverage_files.append(path)
    return coverage_files


def get_data_directories(root: Path) -> List[Tuple[str, Path]]:
    """Get data directories that can be cleaned."""
    data_dir = root / "data"
    dirs = [
        ("processed", data_dir / "processed"),
        ("features", data_dir / "features"),
        ("models", data_dir / "models"),
        ("validation", data_dir / "validation"),
    ]
    return [(name, path) for name, path in dirs if path.exists()]


def get_output_directories(root: Path) -> List[Tuple[str, Path]]:
    """Get output directories that can be cleaned."""
    outputs_dir = root / "outputs"
    if not outputs_dir.exists():
        return []
    
    dirs = []
    for subdir in outputs_dir.iterdir():
        if subdir.is_dir():
            dirs.append((f"outputs/{subdir.name}", subdir))
    return dirs


def remove_paths(paths: List[Path], dry_run: bool = False) -> Tuple[int, int]:
    """
    Remove files and directories.
    
    Returns:
        Tuple of (files_removed, dirs_removed)
    """
    files_removed = 0
    dirs_removed = 0
    
    for path in paths:
        if not path.exists():
            continue
            
        if dry_run:
            if path.is_file():
                print(f"  [DRY RUN] Would remove file: {path}")
                files_removed += 1
            else:
                print(f"  [DRY RUN] Would remove directory: {path}")
                dirs_removed += 1
        else:
            if path.is_file():
                path.unlink()
                files_removed += 1
                print(f"  Removed file: {path}")
            else:
                shutil.rmtree(path)
                dirs_removed += 1
                print(f"  Removed directory: {path}")
    
    return files_removed, dirs_removed


def clean_cache(root: Path, dry_run: bool = False) -> None:
    """Clean Python cache files and directories."""
    print("\n=== Cleaning Python Cache ===")
    
    # __pycache__ directories
    print("\nFinding __pycache__ directories...")
    pycache_dirs = find_pycache_dirs(root)
    if pycache_dirs:
        print(f"Found {len(pycache_dirs)} __pycache__ directories")
        files, dirs = remove_paths(pycache_dirs, dry_run)
        print(f"Cleaned {dirs} directories")
    else:
        print("No __pycache__ directories found")
    
    # .pyc files
    print("\nFinding .pyc files...")
    pyc_files = find_pyc_files(root)
    if pyc_files:
        print(f"Found {len(pyc_files)} .pyc files")
        files, dirs = remove_paths(pyc_files, dry_run)
        print(f"Cleaned {files} files")
    else:
        print("No .pyc files found")


def clean_test_artifacts(root: Path, dry_run: bool = False) -> None:
    """Clean test artifacts."""
    print("\n=== Cleaning Test Artifacts ===")
    
    # pytest cache
    print("\nFinding .pytest_cache directories...")
    pytest_dirs = find_pytest_cache_dirs(root)
    if pytest_dirs:
        print(f"Found {len(pytest_dirs)} .pytest_cache directories")
        files, dirs = remove_paths(pytest_dirs, dry_run)
        print(f"Cleaned {dirs} directories")
    else:
        print("No .pytest_cache directories found")
    
    # mypy cache
    print("\nFinding .mypy_cache directories...")
    mypy_dirs = find_mypy_cache_dirs(root)
    if mypy_dirs:
        print(f"Found {len(mypy_dirs)} .mypy_cache directories")
        files, dirs = remove_paths(mypy_dirs, dry_run)
        print(f"Cleaned {dirs} directories")
    else:
        print("No .mypy_cache directories found")
    
    # coverage files
    print("\nFinding coverage files...")
    coverage_files = find_coverage_files(root)
    if coverage_files:
        print(f"Found {len(coverage_files)} coverage files/directories")
        files, dirs = remove_paths(coverage_files, dry_run)
        print(f"Cleaned {files} files and {dirs} directories")
    else:
        print("No coverage files found")


def clean_notebook_artifacts(root: Path, dry_run: bool = False) -> None:
    """Clean Jupyter notebook artifacts."""
    print("\n=== Cleaning Notebook Artifacts ===")
    
    print("\nFinding .ipynb_checkpoints directories...")
    checkpoint_dirs = find_ipynb_checkpoints(root)
    if checkpoint_dirs:
        print(f"Found {len(checkpoint_dirs)} .ipynb_checkpoints directories")
        files, dirs = remove_paths(checkpoint_dirs, dry_run)
        print(f"Cleaned {dirs} directories")
    else:
        print("No .ipynb_checkpoints directories found")


def clean_logs(root: Path, dry_run: bool = False) -> None:
    """Clean log files."""
    print("\n=== Cleaning Log Files ===")
    
    print("\nFinding log files...")
    log_files = find_log_files(root)
    if log_files:
        print(f"Found {len(log_files)} log files")
        files, dirs = remove_paths(log_files, dry_run)
        print(f"Cleaned {files} files")
    else:
        print("No log files found")


def clean_data(root: Path, dry_run: bool = False, keep_raw: bool = True) -> None:
    """Clean data directories."""
    print("\n=== Cleaning Data Directories ===")
    
    if keep_raw:
        print("(Keeping raw data)")
    
    data_dirs = get_data_directories(root)
    
    if not data_dirs:
        print("No data directories to clean")
        return
    
    for name, path in data_dirs:
        # Skip raw if keeping it
        if keep_raw and name == "raw":
            continue
        
        print(f"\nCleaning data/{name}...")
        # Remove contents but keep the directory (preserve .gitkeep if exists)
        if path.exists():
            gitkeep = path / ".gitkeep"
            has_gitkeep = gitkeep.exists()
            
            if dry_run:
                print(f"  [DRY RUN] Would clean directory: {path}")
                for item in path.iterdir():
                    if item.name != ".gitkeep":
                        print(f"    [DRY RUN] Would remove: {item.name}")
            else:
                items_to_remove = [item for item in path.iterdir() if item.name != ".gitkeep"]
                files, dirs = remove_paths(items_to_remove, dry_run=False)
                print(f"  Cleaned {files} files and {dirs} directories from {name}")
                
                # Ensure .gitkeep exists
                if not has_gitkeep:
                    gitkeep.touch()
                    print(f"  Created .gitkeep")


def clean_outputs(root: Path, dry_run: bool = False) -> None:
    """Clean output directories."""
    print("\n=== Cleaning Output Directories ===")
    
    output_dirs = get_output_directories(root)
    
    if not output_dirs:
        print("No output directories to clean")
        return
    
    for name, path in output_dirs:
        print(f"\nCleaning {name}...")
        if path.exists():
            if dry_run:
                print(f"  [DRY RUN] Would clean directory: {path}")
                for item in path.iterdir():
                    print(f"    [DRY RUN] Would remove: {item.name}")
            else:
                items_to_remove = list(path.iterdir())
                files, dirs = remove_paths(items_to_remove, dry_run=False)
                print(f"  Cleaned {files} files and {dirs} directories from {name}")


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Cleanup utility for defensive-action-expected project"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually removing files"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Clean Python cache files (__pycache__, .pyc)"
    )
    parser.add_argument(
        "--test-artifacts",
        action="store_true",
        help="Clean test artifacts (.pytest_cache, .mypy_cache, coverage)"
    )
    parser.add_argument(
        "--notebooks",
        action="store_true",
        help="Clean notebook artifacts (.ipynb_checkpoints)"
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Clean log files"
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Clean processed data (keeps raw data by default)"
    )
    parser.add_argument(
        "--outputs",
        action="store_true",
        help="Clean output directories"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean everything (cache, test artifacts, notebooks, logs)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Clean everything including data and outputs (USE WITH CAUTION)"
    )
    
    args = parser.parse_args()
    
    # Get project root
    root = get_project_root()
    print(f"Project root: {root}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be removed ***")
    
    # Determine what to clean
    clean_all = args.all or args.full
    clean_full = args.full
    
    # If no specific flags, show help
    if not any([args.cache, args.test_artifacts, args.notebooks, args.logs,
                args.data, args.outputs, args.all, args.full]):
        parser.print_help()
        print("\n⚠ No cleanup options specified. Use --all for standard cleanup.")
        return
    
    # Execute cleanup operations
    if args.cache or clean_all:
        clean_cache(root, args.dry_run)
    
    if args.test_artifacts or clean_all:
        clean_test_artifacts(root, args.dry_run)
    
    if args.notebooks or clean_all:
        clean_notebook_artifacts(root, args.dry_run)
    
    if args.logs or clean_all:
        clean_logs(root, args.dry_run)
    
    if args.data or clean_full:
        clean_data(root, args.dry_run, keep_raw=True)
    
    if args.outputs or clean_full:
        clean_outputs(root, args.dry_run)
    
    print("\n=== Cleanup Complete ===")
    if args.dry_run:
        print("This was a dry run. Run without --dry-run to actually remove files.")


if __name__ == "__main__":
    main()


