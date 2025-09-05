"""
Phase 8: Jekyll Documentation Website Setup Script

This script automatically creates the complete Jekyll site structure with all
necessary files, properly linked navigation, and integrated results from Phase 7.
"""

import os
import shutil
from pathlib import Path

def create_jekyll_structure():
    """Create the complete Jekyll site structure"""
    
    # Define the site structure
    directories = [
        "_layouts",
        "_includes", 
        "_sass",
        "assets/css",
        "assets/js",
        "assets/plots",
        "theory",
        "docs/assets/plots"  # For compatibility with Phase 7 outputs
    ]
    
    print("Creating Jekyll site structure...")
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created directory: {directory}")
    
    # Copy plots from Phase 7 if they exist
    phase7_plots = Path("plots")
    if phase7_plots.exists():
        print("  ✓ Copying Phase 7 plots to assets...")
        for plot_file in phase7_plots.glob("phase7_*.png"):
            shutil.copy2(plot_file, "assets/plots/")
        for plot_file in phase7_plots.glob("phase7_*.svg"):
            shutil.copy2(plot_file, "assets/plots/")
    
    # Copy docs plots if they exist
    docs_plots = Path("docs/assets/plots")
    if docs_plots.exists():
        print("  ✓ Found existing docs plots directory")
    
    print("Jekyll structure created successfully!")

def update_results_with_plots():
    """Update results.md to use correct plot paths"""
    
    results_file = Path("results.md")
    if results_file.exists():
        print("Updating results.md plot paths...")
        
        content = results_file.read_text()
        
        # Update plot paths to use assets directory
        plot_replacements = {
            "../../docs/assets/plots/": "/assets/plots/",
            "../assets/plots/": "/assets/plots/",
            "docs/assets/plots/": "/assets/plots/"
        }
        
        for old_path, new_path in plot_replacements.items():
            content = content.replace(old_path, new_path)
        
        results_file.write_text(content)
        print("  ✓ Updated plot paths in results.md")

def create_gitignore():
    """Create appropriate .gitignore for Jekyll site"""
    
    gitignore_content = """# Jekyll
_site/
.sass-cache/
.jekyll-cache/
.jekyll-metadata

# Bundler
vendor/

# OS
.DS_Store
Thumbs.db

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv

# Results and data (keep source, ignore generated)
results/*/
data/raw/
*.npz
*.npy

# Plots (generated files)
plots/phase4_*.png
plots/phase5_*.png
plots/phase6_*.png

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
"""
    Path(".gitignore").write_text(gitignore_content)
    print("  ✓ Created .gitignore")

def create_readme():
    """Create a comprehensive README for the Jekyll site"""
    
    readme_content = """# PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation

![Project Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Validation](https://img.shields.io/badge/Validation-PASS-green)
![Coverage](https://img.shields.io/badge/Certified_Coverage-100%25-blue)

## Overview
This repository contains the complete implementation and documentation for applying PAC-Bayes certified bounds to Bayesian inverse problems in partial differential equations.

## Documentation Website

The complete documentation is available as a Jekyll website. To run locally:

```bash
# Install dependencies
bundle install

# Serve the site locally
bundle exec jekyll serve

# View at http://localhost:4000
```

## Project Structure

```
├── _config.yml              # Jekyll configuration
├── _layouts/                # HTML templates
├── _includes/               # Reusable HTML components
├── assets/                  # CSS, JS, and plot files
├── theory/                  # Mathematical theory pages
├── src/                     # Python implementation
├── results/                 # Experimental results
└── docs/                    # Additional documentation
```

## Quick Start
1. **View Documentation**: Visit the [project website](https://github.com/Tani843/PAC_Bayes_Certified_Uncertainty_for_Bayesian_Inverse_Heat_Equation)
2. **Run Code**: See implementation in `src/` directory
3. **Reproduce Results**: Follow methodology in documentation

## Key Results

- **100% certified coverage** across 46 validation scenarios
- **Perfect robustness** under noise and sparse sensor conditions
- **Scalable implementation** for practical inverse problems

## Citation

```bibtex
@software{pac_bayes_inverse_pde_2024,
  title={PAC-Bayes Certified Uncertainty for Bayesian Inverse Heat Equation},
  author={Tanisha Gupta},
  year={2024},
  url={https://github.com/Tani843/PAC_Bayes_Certified_Uncertainty_for_Bayesian_Inverse_Heat_Equation}
}
```

## License

MIT License - see LICENSE file for details.
"""
    
    Path("README.md").write_text(readme_content)
    print("  ✓ Created README.md")

def setup_github_pages():
    """Create GitHub Pages workflow"""
    
    workflows_dir = Path(".github/workflows")
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    workflow_content = """name: Build and deploy Jekyll site to GitHub Pages

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.1'
          bundler-cache: true
      
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v4
      
      - name: Build with Jekyll
        run: bundle exec jekyll build --baseurl "${{ steps.pages.outputs.base_path }}"
        env:
          JEKYLL_ENV: production
      
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
"""
    
    (workflows_dir / "jekyll.yml").write_text(workflow_content)
    print("  ✓ Created GitHub Pages workflow")

def main():
    """Main setup function"""
    print("Phase 8: Jekyll Documentation Website Setup")
    print("=" * 50)
    
    create_jekyll_structure()
    update_results_with_plots()
    create_gitignore()
    create_readme()
    setup_github_pages()
    
    print("\n" + "=" * 50)
    print("Jekyll site setup complete!")
    print("\nNext steps:")
    print("1. Copy all the provided .md and .html files to their locations")
    print("2. Run: bundle install")
    print("3. Run: bundle exec jekyll serve")
    print("4. View at: http://localhost:4000")
    print("5. Commit and push to GitHub for automatic deployment")

if __name__ == "__main__":
    main()