"""
GUI for viewing user controls, shortcuts, and mouse interactions.
"""

from html import escape
from typing import Iterable, Tuple

from qtpy import QtWidgets

from sleap.gui.shortcuts import Shortcuts


ControlRows = Iterable[Tuple[str, str]]


class UserControlsDialog(QtWidgets.QDialog):
    """Read-only window listing GUI controls and shortcuts."""

    _ACTION_LABELS = {
        "new": "New Project",
        "open": "Open Project",
        "save": "Save Project",
        "save as": "Save Project As",
        "close": "Quit",
        "add videos": "Add Session",
        "next video": "Next Video",
        "prev video": "Previous Video",
        "goto frame": "Go to Frame",
        "select to frame": "Select to Frame",
        "add instance": "Add Instance",
        "delete instance": "Delete Selected Instance",
        "delete track": "Delete Instance and Track",
        "transpose": "Transpose Instance Tracks",
        "select next": "Select Next Instance",
        "clear selection": "Clear Selection",
        "goto next labeled": "Next Labeled Frame",
        "goto prev labeled": "Previous Labeled Frame",
        "goto last interacted": "Last Interacted Frame",
        "goto next user": "Next User Labeled Frame",
        "goto next suggestion": "Next Suggestion or QC Flag",
        "goto prev suggestion": "Previous Suggestion or QC Flag",
        "goto next track spawn": "Next Track Spawn Frame",
        "show instances": "Show Instances",
        "show labels": "Show Node Names",
        "show edges": "Show Edges",
        "show trails": "Show Trails",
        "color predicted": "Color Predicted Instances",
        "fit": "Fit View to Instances",
        "learning": "Run Training",
        "export clip": "Render Video Clip with Instances",
        "delete frame predictions": "Delete Predictions on Current Frame",
        "delete clip predictions": "Delete Predictions from Clip",
        "delete area predictions": "Delete Predictions from Area",
        "frame next": "Next Frame",
        "frame prev": "Previous Frame",
        "frame next medium step": "Next Frame, Medium Step",
        "frame prev medium step": "Previous Frame, Medium Step",
        "frame next large step": "Next Frame, Large Step",
        "frame prev large step": "Previous Frame, Large Step",
        "export_analysis_current": "Export Analysis HDF5 for Current Video",
    }

    _ADDITIONAL_KEYBOARD_CONTROLS = (
        ("V", "Switch the active video pane/view in multi-video session displays."),
        ("Home", "Go to the first frame."),
        ("End", "Go to the last frame."),
        ("J", "Jump to the start of the contiguous marked region around the frame."),
        ("K", "Jump to the end of the contiguous marked region around the frame."),
        ("1-9", "Select the visible instance at that display index."),
        (
            "Shift + frame navigation",
            "Extend the seekbar frame selection while moving through frames.",
        ),
        ("Ctrl+C", "Copy the selected instance."),
        ("Ctrl+V", "Paste the copied instance."),
        ("Ctrl+Shift+C", "Copy the selected instance track."),
        ("Ctrl+Shift+V", "Paste the copied instance track."),
        ("Ctrl+1 through Ctrl+9", "Assign the selected instance to that track."),
        ("Ctrl+0", "Create a new track."),
        (
            "R over zoomed timeline",
            "Start reach-edit mode when the cursor is over the zoomed timeline.",
        ),
        ("Esc in reach edit", "Cancel reach-edit mode."),
        ("Left/Right in reach edit", "Move the current frame during reach editing."),
    )

    _VIEWER_MOUSE_CONTROLS = (
        ("Mouse wheel", "Zoom the video view in or out."),
        ("Trackpad pinch", "Zoom the video view in or out."),
        ("Left click instance", "Select the instance."),
        ("Left drag node", "Move a user-labeled node and mark it complete."),
        ("Right click node", "Toggle the node as visible or missing."),
        ("Shift + left click node", "Mark all points in that instance as complete."),
        ("Ctrl + left click node/instance", "Duplicate the instance."),
        ("Alt + left drag node", "Move the whole instance from that node."),
        ("Mouse wheel during Alt node drag", "Rotate the instance around that node."),
        ("Ctrl + drag empty frame", "Rubber-band select multiple nodes."),
        ("Drag selected node group", "Move the selected nodes together."),
        (
            "Right click selected node group",
            "Choose Mark Selected Nodes Missing for all selected user nodes.",
        ),
        ("Alt + drag empty frame", "Draw a zoom box and zoom to that rectangle."),
        ("Middle drag", "Pan the video view."),
        ("Left double-click background", "Reset zoom and pan to the full frame."),
        ("Left double-click predicted instance", "Copy prediction to a user instance."),
        (
            "Shift + left double-click predicted instance",
            "Copy prediction to a user instance and mark copied points complete.",
        ),
        (
            "Left double-click user instance",
            "Add any missing skeleton nodes to that instance.",
        ),
        (
            "Right click frame",
            "Open the viewer context menu for placing the selected user instance's "
            "missing nodes or adding instances.",
        ),
        (
            "Right click prediction",
            "Open the viewer context menu with a Delete Prediction option.",
        ),
        ("Drag bounding-box corner", "Resize the instance bounding box."),
        ("Shift + drag inside bounding box", "Move the boxed instance."),
    )

    _SEEKBAR_CONTROLS = (
        ("Click or drag seekbar", "Go to the frame under the pointer."),
        ("Shift + drag seekbar", "Select a frame range."),
        (
            "Shift + double-click seekbar",
            "Select the contiguous marked region around that frame.",
        ),
        ("Alt + drag seekbar", "Zoom the seekbar to the selected frame range."),
        ("Hover seekbar", "Show frame information in a tooltip."),
    )

    _ZOOMED_TIMELINE_CONTROLS = (
        ("Click zoomed timeline", "Go to the clicked frame."),
        ("Hover zoomed timeline", "Show the frame cursor and frame number."),
        ("R", "Start reach-edit mode while hovering the zoomed timeline."),
        ("Esc", "Cancel reach-edit mode."),
        ("Left/Right", "Step the current frame while reach editing."),
    )

    def __init__(
        self,
        shortcuts: Shortcuts = None,
        menu_bar: QtWidgets.QMenuBar = None,
        *args,
        **kwargs,
    ):
        super(UserControlsDialog, self).__init__(*args, **kwargs)

        self.shortcuts = shortcuts or Shortcuts()
        self.menu_bar = menu_bar
        self.setWindowTitle("User Controls")
        self.setModal(False)
        self.resize(780, 680)
        self.make_form()

    def make_form(self):
        """Create the read-only controls window."""
        layout = QtWidgets.QVBoxLayout()

        browser = QtWidgets.QTextBrowser()
        browser.setReadOnly(True)
        browser.setHtml(self._build_html())
        layout.addWidget(browser)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def _build_html(self) -> str:
        """Return the controls guide as simple HTML."""
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: sans-serif; font-size: 12px; }}
                h2 {{ margin-top: 18px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 8px; }}
                th, td {{ border: 1px solid #d0d0d0; padding: 5px 7px; }}
                th {{ background: #eeeeee; text-align: left; }}
                td.control {{ width: 34%; font-weight: 600; }}
                .note {{ color: #555555; }}
            </style>
        </head>
        <body>
            <h1>User Controls</h1>
            <p class="note">
                Keyboard shortcuts reflect the current SLEAP shortcut settings.
                Use Help &gt; Keyboard Shortcuts to edit configurable bindings.
            </p>
            {self._make_shortcuts_section()}
            {self._make_section("Additional Keyboard Controls", self._ADDITIONAL_KEYBOARD_CONTROLS)}
            {self._make_section("Video Viewer Mouse Controls", self._VIEWER_MOUSE_CONTROLS)}
            {self._make_section("Seekbar Controls", self._SEEKBAR_CONTROLS)}
            {self._make_section("Zoomed Timeline Controls", self._ZOOMED_TIMELINE_CONTROLS)}
            {self._make_menu_section()}
        </body>
        </html>
        """

    def _make_shortcuts_section(self) -> str:
        """Return HTML table for configurable keyboard shortcuts."""
        rows = []
        shortcut_items = self.shortcuts[0:len(self.shortcuts)]
        for action, shortcut in shortcut_items.items():
            rows.append(
                (
                    self._ACTION_LABELS.get(action, action.replace("_", " ").title()),
                    self._shortcut_text(shortcut),
                )
            )
        return self._make_section("Configurable Keyboard Shortcuts", rows)

    def _make_section(self, title: str, rows: ControlRows) -> str:
        """Return an HTML table section."""
        table_rows = "\n".join(
            f"<tr><td class='control'>{escape(control)}</td>"
            f"<td>{escape(description)}</td></tr>"
            for control, description in rows
        )
        return f"""
        <h2>{escape(title)}</h2>
        <table>
            <tr><th>Control</th><th>Action</th></tr>
            {table_rows}
        </table>
        """

    def _make_menu_section(self) -> str:
        """Return HTML table for the current main menu commands."""
        if self.menu_bar is None:
            return ""

        rows = []
        for action in self.menu_bar.actions():
            menu = action.menu()
            if menu is None:
                continue
            menu_name = self._clean_action_text(action.text())
            rows.extend(self._iter_menu_actions(menu, menu_name))

        return self._make_section("Main Menu Commands", rows)

    def _iter_menu_actions(self, menu: QtWidgets.QMenu, prefix: str):
        """Yield rows for all actions under a menu."""
        for action in menu.actions():
            if action.isSeparator():
                continue

            action_text = self._clean_action_text(action.text())
            if not action_text:
                continue

            action_path = f"{prefix} > {action_text}"
            submenu = action.menu()
            if submenu is not None:
                yield from self._iter_menu_actions(submenu, action_path)
                continue

            shortcut = action.shortcut().toString()
            yield action_path, shortcut or "No keyboard shortcut"

    @staticmethod
    def _clean_action_text(text: str) -> str:
        """Return menu text without Qt accelerator markers."""
        return text.replace("&", "").strip()

    @staticmethod
    def _shortcut_text(shortcut) -> str:
        """Return display text for a shortcut value."""
        if hasattr(shortcut, "toString"):
            shortcut = shortcut.toString()
        if not shortcut:
            return "Not assigned"
        return str(shortcut)
