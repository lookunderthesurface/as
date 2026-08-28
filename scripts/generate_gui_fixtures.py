"""Generate safe, synthetic GUI fixture images (P3 PPM, no real screenshots).

These fixtures are intentionally zero-secret and tiny: colored bands represent
VSCode/terminal/browser/modal/chart GUIs. The visual pipeline cares about
structure rather than pixels in tests.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "gui"
WIDTH, HEIGHT = 48, 32


def write_ppm(path: Path, rows: list[list[tuple[int, int, int]]]) -> None:
    lines = ["P3", f"{WIDTH} {HEIGHT}", "255"]
    for row in rows:
        for rgb in row:
            lines.append(" ".join(str(v) for v in rgb))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def banded(background: tuple[int, int, int], bands: list[tuple[int, tuple[int, int, int]]]) -> list[list[tuple[int, int, int]]]:
    """Rows where each band (y, color) paints a horizontal stripe."""
    rows: list[list[tuple[int, int, int]]] = []
    for y in range(HEIGHT):
        color = background
        for start, band_color in bands:
            if y >= start:
                color = band_color
        rows.append([color] * WIDTH)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dark = (30, 30, 40)
    grey = (80, 80, 90)
    red = (180, 40, 40)
    green = (40, 160, 80)
    blue = (50, 90, 170)

    # VSCode with failed terminal: dark editor top, red error stripe bottom.
    write_ppm(OUT / "vscode_failed_terminal.ppm", banded(dark, [(6, grey), (20, red), (26, dark)]))
    # VSCode with passing terminal: same layout, green success stripe.
    write_ppm(OUT / "vscode_passed_terminal.ppm", banded(dark, [(6, grey), (20, green), (26, dark)]))
    # Browser documentation: light page with a content band.
    write_ppm(OUT / "browser_docs.ppm", banded((235, 235, 240), [(4, (70, 70, 180)), (10, (200, 220, 240)), (20, (160, 180, 210))]))
    # Modal dialog: mostly dim desktop with a centered bright box.
    rows: list[list[tuple[int, int, int]]] = []
    for y in range(HEIGHT):
        if 12 <= y <= 20:
            rows.append([(245, 245, 250) if 12 <= x <= 34 else dark for x in range(WIDTH)])
        else:
            rows.append([dark] * WIDTH)
    write_ppm(OUT / "modal_dialog.ppm", rows)
    # Chart page: white background with colored column bands.
    chart_rows: list[list[tuple[int, int, int]]] = []
    for y in range(HEIGHT):
        chart_rows.append([(240, 240, 245) if x < 8 or x > 40 else _chart_color(x, y) for x in range(WIDTH)])
    write_ppm(OUT / "chart_page.ppm", chart_rows)
    print(f"wrote fixtures to {OUT}")


def _chart_color(x: int, y: int) -> tuple[int, int, int]:
    bars = [(8, (190, 60, 60)), (14, (70, 140, 200)), (20, (70, 170, 110)), (26, (200, 170, 60)), (32, (150, 90, 180))]
    for start, color in bars:
        if x == start and y > 8:
            return color
    return (250, 250, 250)


if __name__ == "__main__":
    main()
