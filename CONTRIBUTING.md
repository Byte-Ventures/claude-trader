# Contributing to claude-trader

Thank you for your interest in contributing.

This project is **research-oriented and experimental**. It exists to explore automated trading systems, risk controls, and AI-assisted analysis. It does **not** make profit guarantees and is **not** intended as consumer financial software.

## Project Principles

- Safety and risk containment come first.
- Research clarity is preferred over performance optimization.
- Explicit configuration is preferred over implicit behavior.
- Defaults must be conservative.

## What Contributions Are Welcome

- Bug fixes and stability improvements.
- Tests (unit, integration, regression).
- Documentation and clarification of existing behavior.
- Refactors that improve readability or safety without changing behavior.
- Experimental strategy ideas **behind disabled-by-default flags**.
- Improvements to logging, observability, and diagnostics.

## What Is Out of Scope

- Profit claims, marketing language, or performance promises.
- Signal selling, copy-trading, or monetization features.
- Changes that bypass exchange safeguards or ToS.
- Hard-coded credentials, secrets, or exchange-specific hacks.
- Features that assume live trading without explicit user opt-in.

## Safety Requirements

- Any change that affects **order execution, position sizing, or risk limits** must:
  - Default to safe behavior.
  - Be clearly documented.
  - Be covered by tests where feasible.
- New features should assume **paper trading** unless explicitly enabled.
- Breaking changes must be clearly called out in the PR description.

## Contribution Process

1. Fork the repository.
2. Create a feature branch from `develop`:
   ```bash
   git checkout -b feature/my-change develop
   ```
3. Make your changes and update documentation if needed.
4. Run tests locally.
5. Open a Pull Request targeting `develop`.
6. Describe **what changed**, **why**, and **risk implications**.

Link to a related Issue or Discussion if applicable.

## Code Style and Workflow

- Follow existing structure and conventions.
- Keep commits focused and readable.
- Versioning and branching rules are documented in `CLAUDE.md`.

## Disclaimer

By contributing, you acknowledge that this is **experimental software** and that users may lose money. Contributions must not imply financial advice or guarantees.

---

If you are unsure whether a change fits the project’s scope, open a Discussion first.
