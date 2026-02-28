# Watanabe's Singular Learning Theory

This repository contains notes and computational explorations based on Sumio Watanabe's book *"Mathematical Theory of Bayesian Statistics"* (often referred to as Singular Learning Theory).

The project is built as a [Quarto](https://quarto.org/) website.

## Contents

- **[Cheat Sheet](cheat_sheet.qmd)**: Important Definitions and Relations in Singular Learning Theory.
- **[Regular and Singular Models](regular_and_singular_models.qmd)**: Notes and code reproducing Figures 1.2 to 1.5 from the book.
- **[SLT Gaussian Mixture](slt_gaussian_mixture.qmd)**: An illustration of Watanabe's Singular Learning Theory using a Gaussian Mixture Model.

## Running Locally

To build and preview the website locally, you will need to install Quarto. You may also need Python and the relevant scientific libraries (like `numpy`, `scipy`, `matplotlib`) to execute any embedded code blocks in the `.qmd` files.

Once Quarto is installed, you can preview the website by running:

```bash
quarto preview
```

To render the static HTML website, run:

```bash
quarto render
```

The generated files will be placed in the `_site/` directory.

## License

This repository is for educational purposes and personal notes.
