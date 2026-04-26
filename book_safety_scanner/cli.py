"""CLI entry point."""

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from .database import ScanDatabase
from .ingestion import parse_epub
from .output import build_manifest, save_html_report, save_manifest
from .pipeline import assemble_analyses, run_pass1, run_pass2, run_pass3
from .redactor import generate_bridge, get_context_texts, rewrite_epub
from .scoring import compute_book_profile, compute_chapter_scores, merge_skip_regions
from .taxonomy import AGE_BANDS, CATEGORY_LABELS

app = typer.Typer(
    name="book-safety",
    help="Scan books for content not suitable for young children.",
    no_args_is_help=True,
    invoke_without_command=False,
)
console = Console()


def _progress_bar(description: str, total: int):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    )
    task = progress.add_task(description, total=total)
    return progress, task


@app.command()
def scan(
    epub_path: Annotated[Path, typer.Argument(help="Path to the EPUB file")],
    age_band: Annotated[
        str,
        typer.Option(
            "--age-band",
            help="Age band: under_7 | 7_10 | 10_12 | 12_plus",
        ),
    ] = "10_12",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output JSON manifest path"),
    ] = Path("scan_results.json"),
    html: Annotated[
        bool,
        typer.Option("--html/--no-html", help="Also generate an HTML report"),
    ] = True,
    pass1_only: Annotated[
        bool,
        typer.Option("--pass1-only", help="Run keyword filter only (no LLM calls)"),
    ] = False,
    skip_pass3: Annotated[
        bool,
        typer.Option("--skip-pass3", help="Skip chapter-level coherence check"),
    ] = False,
    db_path: Annotated[
        Path,
        typer.Option("--db", help="SQLite cache database path"),
    ] = Path("scan_cache.db"),
    request_delay: Annotated[
        float,
        typer.Option("--delay", help="Seconds to wait between LLM requests"),
    ] = 0.3,
):
    """Scan an EPUB for child-inappropriate content."""

    if age_band not in AGE_BANDS:
        console.print(
            f"[red]Unknown age band '{age_band}'. "
            f"Choose from: {', '.join(AGE_BANDS)}"
        )
        raise typer.Exit(1)

    if not epub_path.exists():
        console.print(f"[red]File not found: {epub_path}")
        raise typer.Exit(1)

    console.rule("[bold]Book Safety Scanner")

    # ── Ingest ────────────────────────────────────────────────────────────────
    with ScanDatabase(db_path) as db:
        if db.has_paragraphs():
            console.print("[dim]Resuming from cache...[/dim]")
            paragraphs = db.load_paragraphs()
            # Re-parse meta (cheap)
            meta, _ = parse_epub(epub_path)
        else:
            console.print(f"Parsing [bold]{epub_path.name}[/bold]...")
            meta, paragraphs = parse_epub(epub_path)
            db.insert_paragraphs(paragraphs)

        console.print(
            f"  [green]✓[/green] {meta.title} by {meta.author} "
            f"— {meta.total_paragraphs:,} paragraphs, {meta.total_words:,} words"
        )

        # ── Pass 1 ────────────────────────────────────────────────────────────
        console.print("\n[bold]Pass 1[/bold]: keyword filter")
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Scanning keywords...", total=meta.total_paragraphs)

            def p1_cb(done, total):
                progress.update(task, completed=done)

            candidate_ids = run_pass1(paragraphs, db, p1_cb)

        console.print(
            f"  [green]✓[/green] {len(candidate_ids):,} candidate paragraphs "
            f"({100 * len(candidate_ids) / max(1, meta.total_paragraphs):.1f}%)"
        )

        if pass1_only:
            console.print("[yellow]--pass1-only: stopping after Pass 1[/yellow]")
            _print_pass1_summary(paragraphs, candidate_ids, db)
            raise typer.Exit(0)

        # ── Pass 2 ────────────────────────────────────────────────────────────
        already_analyzed = db.get_analyzed_ids()
        todo_count = len(candidate_ids - already_analyzed)
        console.print(
            f"\n[bold]Pass 2[/bold]: LLM analysis "
            f"({todo_count} to analyse, {len(already_analyzed)} cached)"
        )

        if todo_count == 0:
            console.print("  [dim]All candidates already cached[/dim]")
            llm_results = db.load_llm_results()
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Analysing with LLM...", total=todo_count)

                def p2_cb(done, total):
                    progress.update(task, completed=done)

                llm_results = run_pass2(
                    paragraphs, candidate_ids, db, p2_cb, request_delay
                )

        flagged_count = sum(
            1 for r in llm_results.values()
            if any(r.get(cat, {}).get("score", 0) > 0 for cat in CATEGORY_LABELS)
        )
        console.print(f"  [green]✓[/green] {flagged_count} paragraphs with non-zero scores")

        # ── Pass 3 ────────────────────────────────────────────────────────────
        if not skip_pass3:
            unflagged_chapters = _count_unflagged_chapters(paragraphs, llm_results)
            already_ch = len(db.get_analyzed_chapters())
            todo_ch = unflagged_chapters - already_ch
            console.print(
                f"\n[bold]Pass 3[/bold]: chapter coherence "
                f"({todo_ch} chapters to check)"
            )

            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Checking chapters...", total=max(todo_ch, 1))

                def p3_cb(done, total):
                    progress.update(task, completed=done, total=max(total, 1))

                chapter_results = run_pass3(paragraphs, llm_results, db, p3_cb)
        else:
            chapter_results = db.load_chapter_results()

        # ── Scoring & output ──────────────────────────────────────────────────
        analyses = assemble_analyses(paragraphs, candidate_ids, llm_results, db)
        chapter_scores = compute_chapter_scores(paragraphs, analyses)
        skip_regions = merge_skip_regions(paragraphs, analyses, age_band)
        book_profile = compute_book_profile(analyses)
        book_profile.total_skip_regions = len(skip_regions)

        manifest = build_manifest(meta, age_band, skip_regions, chapter_scores, book_profile)
        save_manifest(manifest, output)
        console.print(f"\n[green]✓[/green] Manifest saved to [bold]{output}[/bold]")

        if html:
            html_path = output.with_suffix(".html")
            save_html_report(manifest, html_path)
            console.print(f"[green]✓[/green] HTML report saved to [bold]{html_path}[/bold]")

        _print_final_summary(manifest)


def _count_unflagged_chapters(paragraphs, llm_results) -> int:
    from .taxonomy import CATEGORIES
    flagged_chapters = {
        p.chapter_num
        for p in paragraphs
        if p.id in llm_results
        and any(llm_results[p.id].get(cat, {}).get("score", 0) > 0 for cat in CATEGORIES)
    }
    all_chapters = {p.chapter_num for p in paragraphs}
    return len(all_chapters - flagged_chapters)


def _print_pass1_summary(paragraphs, candidate_ids, db):
    from .taxonomy import CATEGORY_LABELS
    table = Table(title="Pass 1 Keyword Hits by Category")
    table.add_column("Category")
    table.add_column("Paragraphs Flagged", justify="right")

    cat_counts: dict[str, int] = {c: 0 for c in CATEGORY_LABELS}
    for pid in candidate_ids:
        for cat in db.get_keyword_categories(pid):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    for cat, label in CATEGORY_LABELS.items():
        if cat_counts.get(cat, 0) > 0:
            table.add_row(label, str(cat_counts[cat]))

    console.print(table)


def _print_final_summary(manifest: dict):
    console.rule("[bold]Results")

    profile = manifest["book_profile"]
    table = Table(title="Book Safety Profile")
    table.add_column("Category")
    table.add_column("Mean Score", justify="right")

    for cat, score in profile.items():
        colour = "green" if score < 1 else "yellow" if score < 2.5 else "red"
        table.add_row(cat, f"[{colour}]{score:.2f}[/{colour}]")

    console.print(table)

    skip_regions = manifest["skip_regions"]
    if skip_regions:
        table2 = Table(title=f"{len(skip_regions)} Skip Regions")
        table2.add_column("Ch.")
        table2.add_column("Title")
        table2.add_column("Para range")
        table2.add_column("Severity", justify="right")
        table2.add_column("Categories")

        for sr in skip_regions:
            sev = sr["max_severity"]
            colour = "yellow" if sev < 3 else "red"
            table2.add_row(
                str(sr["chapter"]),
                sr["chapter_title"][:40],
                f"{sr['paragraph_range'][0]}–{sr['paragraph_range'][1]}",
                f"[{colour}]{sev:.1f}[/{colour}]",
                ", ".join(sr["categories"]),
            )
        console.print(table2)
    else:
        console.print(
            "[green]No passages exceeded the age-band threshold for this book.[/green]"
        )


@app.command()
def redact(
    epub_path: Annotated[Path, typer.Argument(help="Path to the original EPUB file")],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", "-m", help="Scan manifest JSON produced by the scan command"),
    ] = Path("scan_results.json"),
    db: Annotated[
        Path,
        typer.Option("--db", help="SQLite cache database from the scan"),
    ] = Path("scan_cache.db"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for the redacted EPUB"),
    ] = Path(""),
) -> None:
    """Generate a child-safe EPUB by replacing flagged passages with Sonnet-written bridges."""

    if not epub_path.exists():
        console.print(f"[red]EPUB not found: {epub_path}")
        raise typer.Exit(1)
    if not manifest.exists():
        console.print(f"[red]Manifest not found: {manifest} — run the scan command first")
        raise typer.Exit(1)
    if not db.exists():
        console.print(f"[red]Database not found: {db} — run the scan command first")
        raise typer.Exit(1)

    import json
    with open(manifest) as f:
        manifest_data = json.load(f)

    skip_regions = manifest_data.get("skip_regions", [])
    if not skip_regions:
        console.print("[green]No skip regions in manifest — nothing to redact.")
        raise typer.Exit(0)

    out_path = output if str(output) else epub_path.with_stem(epub_path.stem + "_childsafe")
    console.rule("[bold]Book Redactor")
    console.print(
        f"  {len(skip_regions)} skip regions to bridge  |  "
        f"output: [bold]{out_path}[/bold]"
    )

    bridges: list[str] = []

    with ScanDatabase(db) as scan_db:
        cached_bridges = scan_db.load_bridges()
        todo = sum(
            1 for sr in skip_regions
            if tuple(sr["paragraph_range"]) not in cached_bridges
        )
        cached_count = len(skip_regions) - todo
        if cached_count:
            console.print(f"  [dim]{cached_count} bridges already cached[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Generating bridges with Sonnet ({todo} new)...",
                total=len(skip_regions),
            )

            for sr in skip_regions:
                start_id, end_id = sr["paragraph_range"]
                key = (start_id, end_id)

                if key in cached_bridges:
                    bridges.append(cached_bridges[key])
                else:
                    before_texts, after_texts = get_context_texts(db, start_id, end_id)
                    bridge = generate_bridge(
                        summary=sr["summary"],
                        before_texts=before_texts,
                        after_texts=after_texts,
                    )
                    scan_db.insert_bridge(start_id, end_id, bridge)
                    bridges.append(bridge)

                progress.advance(task)

    console.print(f"  [green]✓[/green] {len(bridges)} bridges ready")

    console.print("  Rewriting EPUB...")
    rewrite_epub(epub_path, skip_regions, bridges, out_path)
    console.print(f"[green]✓[/green] Saved to [bold]{out_path}[/bold]")

    # Show what was written
    table = Table(title="Bridges Written")
    table.add_column("Ch.", style="dim")
    table.add_column("Severity", justify="right")
    table.add_column("Bridge text")

    for sr, bridge in zip(skip_regions, bridges):
        sev = sr["max_severity"]
        colour = "yellow" if sev < 3 else "red"
        table.add_row(
            str(sr["chapter"]),
            f"[{colour}]{sev:.1f}[/{colour}]",
            bridge[:90] + ("…" if len(bridge) > 90 else ""),
        )
    console.print(table)
