"""
Cleanup script to remove redundant/obsolete markdown documentation files.

This script identifies and removes markdown files that:
1. Are transitional/implementation guides from completed work
2. Have been superseded by newer documentation
3. Contain outdated status information
4. Are redundant with other documentation

Run with --dry-run to preview changes before applying.
"""

import argparse
from pathlib import Path
import shutil
from typing import List, Tuple


def get_files_to_remove() -> List[Tuple[Path, str]]:
    """
    Returns list of (file_path, reason) tuples for files to remove.
    
    Returns:
        List of (Path, str) tuples where Path is file to remove and str is reason
    """
    project_root = Path(__file__).parent.parent.parent
    
    files_to_remove = [
        # Transitional implementation guides (xT switch completed)
        (
            project_root / "IMPLEMENTATION.md",
            "Old implementation summary from early pipeline phase, superseded by current docs"
        ),
        (
            project_root / "IMPLEMENTATION_STEPS.md",
            "Step-by-step xT implementation guide, work completed, kept in git history"
        ),
        (
            project_root / "SWITCH_TO_XT_SUMMARY.md",
            "Transitional summary for xT switch, work completed"
        ),
        (
            project_root / "README_XT_IMPLEMENTATION.md",
            "Complete xT implementation package, work completed and integrated"
        ),
        
        # Outdated status files
        (
            project_root / "PIPELINE_STATUS.md",
            "Old pipeline status from June 8, 2026, no longer actively updated"
        ),
        
        # Redundant cleanup documentation
        (
            project_root / "docs" / "system" / "CLEANUP_SUMMARY.md",
            "Redundant with CLEANUP_IMPLEMENTATION.md, less comprehensive"
        ),
        
        # Redundant docs/XT_TARGET_IMPLEMENTATION.md (if exists)
        (
            project_root / "docs" / "XT_TARGET_IMPLEMENTATION.md",
            "Duplicate of root level xT implementation docs"
        ),
    ]
    
    # Filter to only existing files
    return [(f, r) for f, r in files_to_remove if f.exists()]


def remove_files(dry_run: bool = False) -> None:
    """
    Remove identified files.
    
    Args:
        dry_run: If True, only print what would be removed without actually removing
    """
    files_to_remove = get_files_to_remove()
    
    if not files_to_remove:
        print("✓ No files to remove - workspace is clean!")
        return
    
    print(f"{'DRY RUN: ' if dry_run else ''}Found {len(files_to_remove)} obsolete markdown files:\n")
    
    total_size = 0
    for file_path, reason in files_to_remove:
        size = file_path.stat().st_size
        total_size += size
        size_kb = size / 1024
        
        # Get relative path for display
        rel_path = file_path.relative_to(file_path.parent.parent.parent)
        
        print(f"  - {rel_path}")
        print(f"    Size: {size_kb:.1f} KB")
        print(f"    Reason: {reason}")
        print()
    
    print(f"Total size to free: {total_size / 1024:.1f} KB\n")
    
    if dry_run:
        print("DRY RUN: No files were actually removed.")
        print("Run without --dry-run to apply changes.\n")
        return
    
    # Actually remove files
    removed_count = 0
    for file_path, reason in files_to_remove:
        try:
            file_path.unlink()
            removed_count += 1
            print(f"✓ Removed: {file_path.name}")
        except Exception as e:
            print(f"✗ Failed to remove {file_path.name}: {e}")
    
    print(f"\n✓ Successfully removed {removed_count} of {len(files_to_remove)} files.")
    print(f"✓ Freed {total_size / 1024:.1f} KB of disk space.")


def list_remaining_docs() -> None:
    """List markdown files that will remain after cleanup."""
    project_root = Path(__file__).parent.parent.parent
    
    # Important docs to keep
    keep_docs = [
        # Root level - essential project documentation
        ("README.md", "Main project README"),
        ("NOTEBOOKS_QUICKSTART.md", "Notebook usage guide"),
        ("XT_REGRESSION_QUICK_REFERENCE.md", "Quick reference for xT regression modeling"),
        ("CLEANUP_QUICK_REFERENCE.md", "Cleanup commands quick reference"),
        
        # docs/ root - core documentation
        ("docs/README.md", "Documentation index"),
        ("docs/project_scope.md", "Project scope and definition"),
        ("docs/implementation_plan.md", "Development roadmap"),
        ("docs/player_defense_model.md", "Player defensive action modeling"),
        ("docs/possession_sequences.md", "Possession-level features"),
        ("docs/possession_visualization.md", "Visualization guide"),
        ("docs/notebook_findings_summary.md", "Notebook analysis results"),
        ("docs/notebook_remediation_plan.md", "Notebook remediation steps"),
        ("docs/BASELINE_MODELING_STRATEGY.md", "Complete modeling strategy"),
        ("docs/MODELING_QUICK_REFERENCE.md", "1-page modeling summary"),
        ("docs/MODELING_PROGRESS.md", "Modeling progress tracking"),
        
        # docs/analysis/ - analytical findings
        ("docs/analysis/README.md", "Analysis documentation index"),
        ("docs/analysis/ZONE_ANALYSIS.md", "12×8 grid shot analysis"),
        ("docs/analysis/COUNTER_ATTACK_EXPLANATION.md", "Counter-attack mechanics"),
        ("docs/analysis/INSIGHT_VALIDATION.md", "Tactical insights validation"),
        ("docs/analysis/XT_BASED_TARGET_ANALYSIS.md", "xT target comprehensive analysis"),
        ("docs/analysis/baseline_model_results.md", "Baseline model results"),
        ("docs/analysis/MODEL_PORTFOLIO_REPORT.md", "Complete model portfolio comparison"),
        
        # docs/system/ - system documentation
        ("docs/system/README.md", "System architecture overview"),
        ("docs/system/PIPELINE_FIX_SUMMARY.md", "Critical bug fixes"),
        ("docs/system/REPO_NAVIGATION.md", "Repository structure guide"),
        ("docs/system/CLEANUP_IMPLEMENTATION.md", "Cleanup procedure implementation"),
    ]
    
    print("\n" + "="*80)
    print("DOCUMENTATION STRUCTURE AFTER CLEANUP")
    print("="*80 + "\n")
    
    current_section = None
    for file_path, description in keep_docs:
        # Determine section
        if file_path.startswith("docs/analysis/"):
            section = "docs/analysis/"
        elif file_path.startswith("docs/system/"):
            section = "docs/system/"
        elif file_path.startswith("docs/"):
            section = "docs/"
        else:
            section = "root"
        
        # Print section header if changed
        if section != current_section:
            print(f"\n{section.upper() if section != 'root' else 'PROJECT ROOT'}")
            print("-" * 40)
            current_section = section
        
        # Check if file exists
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path.split('/')[-1]:<40} {description}")
        else:
            print(f"  ⚠ {file_path.split('/')[-1]:<40} {description} [MISSING]")
    
    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Remove obsolete markdown documentation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be removed
  python scripts/utils/cleanup_docs.py --dry-run
  
  # Remove obsolete files
  python scripts/utils/cleanup_docs.py
  
  # Show remaining documentation structure
  python scripts/utils/cleanup_docs.py --list-remaining
        """
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without actually removing files"
    )
    
    parser.add_argument(
        "--list-remaining",
        action="store_true",
        help="Show documentation structure that will remain after cleanup"
    )
    
    args = parser.parse_args()
    
    if args.list_remaining:
        list_remaining_docs()
    else:
        remove_files(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

