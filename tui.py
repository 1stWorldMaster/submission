from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, OptionList
from textual.screen import Screen
from textual.widgets.option_list import Option

"""
Menu app that **automatically exits** after the user makes a final (leaf) selection.

The chosen leaf label is stored in ``app.selected_option`` and printed once the
Textual event loop ends.
"""

# -------------------------------------------------------------
# Helper: pair each label with an optional sub‑menu list
# -------------------------------------------------------------

def _pair_items(raw_items: list[str | list]) -> list[tuple[str, list | None]]:
    """Convert a *flat* list into ``(label, sub_list | None)`` tuples."""
    pairs: list[tuple[str, list | None]] = []
    i = 0
    while i < len(raw_items):
        label = raw_items[i]
        if not isinstance(label, str):
            raise ValueError("A sub‑menu list cannot precede its parent label.")

        sub: list | None = None
        if i + 1 < len(raw_items) and isinstance(raw_items[i + 1], list):
            sub = raw_items[i + 1]
            i += 1  # Skip the sub‑list we just consumed

        pairs.append((label, sub))
        i += 1
    return pairs


# -------------------------------------------------------------
# Recursive Menu screen
# -------------------------------------------------------------

class MenuScreen(Screen):
    """A screen that can show a (sub‑)menu at any depth."""

    BINDINGS = [("q", "app.pop_screen", "Back")]

    def __init__(
        self,
        items: list[str | list],
        *,
        title: str = "Menu",
        show_back: bool = True,
    ) -> None:
        super().__init__()
        self._pairs = _pair_items(items)
        self._title = title
        self._show_back = show_back

    # ----- layout -------------------------------------------
    def compose(self) -> ComposeResult:  # noqa: D401 – Textual API
        yield Header()
        yield Static(f"{self._title} – Use ↑ ↓ and Enter", classes="title")

        option_widgets = [
            Option(label, id=str(idx)) for idx, (label, _) in enumerate(self._pairs)
        ]
        if self._show_back:
            option_widgets.append(Option("Back", id="back"))

        self.options = OptionList(*option_widgets)
        yield self.options
        yield Static("", id="status_message")
        yield Footer()

    # ----- interaction --------------------------------------
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:  # noqa: D401
        sel = event.option

        # Handle automatic "Back" in sub‑menus
        if sel.id == "back":
            self.app.pop_screen()
            return

        idx = int(sel.id)
        label, sub = self._pairs[idx]

        if sub:
            # Deeper level → push new MenuScreen
            self.app.push_screen(MenuScreen(sub, title=label, show_back=True))
        else:
            # Leaf item chosen – store & exit immediately
            self.app.selected_option = label
            self.app.exit()


# -------------------------------------------------------------
# Main Textual application
# -------------------------------------------------------------

class MenuApp(App):
    """Display nested menus; quit after the first leaf choice."""

    CSS = """
    Screen {
        align: center middle;
        background: #1e1e2e;
        color: #dcdcdc;
    }

    .title {
        content-align: center middle;
        padding: 1 0;
        color: #89dceb;
        text-style: bold;
    }

    OptionList {
        width: 40;
        border: round #44475a;
        background: #282a36;
        margin: 2 0;
        padding: 1;
    }

    OptionList > .option {
        padding: 1 2;
        color: #f8f8f2;
    }

    OptionList > .option--highlighted {
        background: #6272a4;
        color: #ffffff;
        text-style: bold;
    }

    Header {
        background: #44475a;
        color: #f8f8f2;
        text-style: bold;
    }

    Footer {
        background: #44475a;
        color: #bd93f9;
    }

    #status_message {
        content-align: center middle;
        padding: 1;
        color: #50fa7b;
        text-style: bold;
    }
    """

    def __init__(self, menu_data: list[str | list]):
        super().__init__()
        self._menu_data = menu_data
        self.selected_option: str | None = None  # Will hold the final choice

    def on_mount(self) -> None:
        # Root screen has *no* Back button
        self.push_screen(MenuScreen(self._menu_data, title="Main Menu", show_back=False))


# -------------------------------------------------------------
# Example usage
# -------------------------------------------------------------

if __name__ == "__main__":
    menu_data = [
        "Option 1",
        "Option 2",
        [
            "Option 2.1",
            "Option 2.2",
        ],
        "Option 3",
        [
            "Option 3.1",
        ],
        "Exit",  # Leaf item – still quits immediately
    ]

    app = MenuApp(menu_data)
    app.run()

    print("Selected option:", app.selected_option)

