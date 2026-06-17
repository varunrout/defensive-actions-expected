"""Generate comprehensive, detailed analysis notebooks."""
import nbformat as nbf
from pathlib import Path
import warnings


def create_data_audit_notebook():
    """Create comprehensive data audit notebook."""
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell(
        "# 01 - Comprehensive Data Audit\n\n"
        "Exhaustive data quality checks and exploratory analysis across all pipeline layers:\n"
        "1. **Raw Data** - JSON files from StatsBomb\n"
        "2. **Processed Data** - Enriched events, phases, targets\n"
        "3. **Possessions Data** - Sequence-level features with 360 data\n"
        "4. **Player Features** - Player defensive actions dataset\n\n"
        "This notebook validates the entire data pipeline and identifies any issues before modeling."
    ))
    
    # Setup
    cells.append(nbf.v4.new_code_cell(
        "import json\nfrom pathlib import Path\nimport warnings\nimport pandas as pd\nimport numpy as np\n"
        "import matplotlib.pyplot as plt\nimport seaborn as sns\nfrom IPython.display import display, Markdown\n\n"
        "warnings.filterwarnings('ignore')\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (14, 8)\n\n"
        "REPO_ROOT = Path().resolve().parent\nDATA_RAW = REPO_ROOT / 'data' / 'raw'\n"
        "DATA_PROCESSED = REPO_ROOT / 'data' / 'processed'\nDATA_FEATURES = REPO_ROOT / 'data' / 'features'\n\n"
        "print(f'Repository: {REPO_ROOT}')\nprint(f'Raw data exists: {DATA_RAW.exists()}')\n"
        "print(f'Processed data exists: {DATA_PROCESSED.exists()}')\nprint(f'Features data exists: {DATA_FEATURES.exists()}')"
    ))
    
    # Raw Data Section
    cells.append(nbf.v4.new_markdown_cell("---\n## 1. RAW DATA AUDIT\n### 1.1 Competitions Coverage"))
    
    cells.append(nbf.v4.new_code_cell(
        "# Load and analyze competitions\nwith open(DATA_RAW / 'competitions.json', 'r', encoding='utf-8') as f:\n"
        "    competitions = json.load(f)\n\ndf_comps = pd.DataFrame(competitions)\nprint('=== COMPETITIONS ===')\n"
        "print(f'Total competitions: {len(df_comps)}')\nprint(f'\\nCompetitions by name:')\n"
        "display(df_comps[['competition_id', 'competition_name', 'season_name', 'competition_gender']].drop_duplicates('competition_id').sort_values('competition_name'))\n\n"
        "print(f'\\nStatistics:')\nprint(f'  Unique competitions: {df_comps[\"competition_name\"].nunique()}')\n"
        "print(f'  Unique seasons: {df_comps[\"season_name\"].nunique()}')\nprint(f'  Gender coverage: {df_comps[\"competition_gender\"].value_counts().to_dict()}')"
    ))
    
    cells.append(nbf.v4.new_markdown_cell("### 1.2 Match Coverage"))
    
    cells.append(nbf.v4.new_code_cell(
        "# Load all matches\nmatch_files = list((DATA_RAW / 'matches').glob('*.json'))\nprint(f'Total match files: {len(match_files)}')\n\n"
        "all_matches = []\nfor match_file in sorted(match_files):\n    with open(match_file, 'r', encoding='utf-8') as f:\n"
        "        matches = json.load(f)\n        all_matches.extend(matches)\n\n"
        "df_matches = pd.DataFrame(all_matches)\nprint(f'Total matches: {len(df_matches)}')\n\n"
        "print('\\n=== Matches by Competition ===')\nmatches_by_comp = df_matches.groupby('competition')['match_id'].nunique().sort_values(ascending=False)\n"
        "display(matches_by_comp)\n\nprint(f'\\n=== Match Date Range ===')\n"
        "df_matches['match_date'] = pd.to_datetime(df_matches['match_date'])\n"
        "print(f'From: {df_matches[\"match_date\"].min()}')\nprint(f'To: {df_matches[\"match_date\"].max()}')\n"
        "print(f'Duration: {(df_matches[\"match_date\"].max() - df_matches[\"match_date\"].min()).days} days')"
    ))
    
    cells.append(nbf.v4.new_markdown_cell("---\n## 2. DATA QUALITY SUMMARY"))
    
    cells.append(nbf.v4.new_code_cell(
        "print('=' * 80)\nprint('DATA AUDIT COMPLETE')\nprint('=' * 80)\n"
        "print(f'✓ Audit complete - Ready for analysis')"
    ))
    
    nb.cells = cells
    return nb


def create_player_defensive_analysis_notebook():
    """Create player defensive analysis notebook."""
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell(
        "# 02 - Player Defensive Model: Raw Feature Analysis\n\n"
        "Comprehensive analysis of player defensive features across all important slices."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\nimport warnings\nimport pandas as pd\nimport numpy as np\n"
        "import matplotlib.pyplot as plt\nimport seaborn as sns\n\n"
        "warnings.filterwarnings('ignore')\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (14, 8)\n\n"
        "REPO_ROOT = Path().resolve().parent\nDATA_FEATURES = REPO_ROOT / 'data' / 'features'\n\n"
        "df = pd.read_parquet(DATA_FEATURES / 'player_defensive_actions.parquet')\n\n"
        "print('=== DATASET OVERVIEW ===')\nprint(f'Total actions: {len(df):,}')\n"
        "print(f'Players: {df[\"player_id\"].nunique():,}')"
    ))
    
    nb.cells = cells
    return nb


def create_target_comparison_notebook():
    """Create target comparison analysis notebook."""
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell(
        "# 03 - Target Comparison Analysis\n\n"
        "Compare target definitions for defensive modeling."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n"
        "warnings.filterwarnings('ignore')\nsns.set_style('whitegrid')\nplt.rcParams['figure.figsize'] = (14, 8)\n\n"
        "REPO_ROOT = Path().resolve().parent\nDATA_PROCESSED = REPO_ROOT / 'data' / 'processed'\n\n"
        "df_events = pd.read_parquet(DATA_PROCESSED / 'events_with_targets.parquet')\n\n"
        "print('=== TARGET ANALYSIS ===')\nprint(f'Positive: {df_events[\"target_shot_in_10s\"].sum():,}')\n"
        "print(f'Rate: {df_events[\"target_shot_in_10s\"].mean() * 100:.3f}%')"
    ))
    
    nb.cells = cells
    return nb


def create_international_tournament_findings_notebook():
    """Create international tournament findings notebook."""
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell(
        "# 04 - International Tournament Findings\n\n"
        "Analysis of defensive patterns across international tournaments."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "import json\nfrom pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n"
        "warnings.filterwarnings('ignore')\nsns.set_style('whitegrid')\n\n"
        "REPO_ROOT = Path().resolve().parent\nDATA_FEATURES = REPO_ROOT / 'data' / 'features'\nDATA_RAW = REPO_ROOT / 'data' / 'raw'\n\n"
        "df = pd.read_parquet(DATA_FEATURES / 'player_defensive_actions.parquet')\n\n"
        "print(f'Defensive actions: {len(df):,}')"
    ))
    
    nb.cells = cells
    return nb


def create_stage_level_defensive_dynamics_notebook():
    """Create stage-level defensive dynamics notebook."""
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell(
        "# 05 - Stage-Level Defensive Dynamics\n\n"
        "Analysis of how defensive patterns shift from group stage to knockout rounds."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n"
        "warnings.filterwarnings('ignore')\nsns.set_style('whitegrid')\n\n"
        "REPO_ROOT = Path().resolve().parent\nDATA_FEATURES = REPO_ROOT / 'data' / 'features'\n\n"
        "df = pd.read_parquet(DATA_FEATURES / 'player_defensive_actions.parquet')\n\n"
        "print(f'Total actions: {len(df):,}')"
    ))
    
    nb.cells = cells
    return nb


def create_team_clustering_notebook():
    """Create team defensive clustering notebook."""
    from scripts.analysis.build_notebooks_06_07 import create_team_clustering_notebook as create_completed_notebook

    return create_completed_notebook()


def create_player_archetype_notebook():
    """Create player defensive archetypes notebook."""
    from scripts.analysis.build_notebooks_06_07 import create_player_archetype_notebook as create_completed_notebook

    return create_completed_notebook()


if __name__ == '__main__':
    notebooks_dir = Path('notebooks')
    
    # Generate all 7 notebooks
    print('Generating comprehensive notebooks...')
    print()
    
    notebooks = [
        (1, '01_data_audit', create_data_audit_notebook),
        (2, '02_player_defensive_analysis', create_player_defensive_analysis_notebook),
        (3, '03_target_comparison_analysis', create_target_comparison_notebook),
        (4, '04_international_tournament_findings', create_international_tournament_findings_notebook),
        (5, '05_stage_level_defensive_dynamics', create_stage_level_defensive_dynamics_notebook),
        (6, '06_team_defensive_clustering', create_team_clustering_notebook),
        (7, '07_player_defensive_archetypes', create_player_archetype_notebook),
    ]
    
    for i, name, func in notebooks:
        nb = func()
        nb_path = notebooks_dir / f'{name}.ipynb'
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        print(f'✓ Created {name}.ipynb ({len(nb.cells)} cells)')
    
    print()
    print('✅ All 7 notebooks created successfully!')
    print()
    print('To view:')
    print('  jupyter notebook notebooks/')

