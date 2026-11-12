"""Regression tests for theme placement, UI styling and sidebar controls."""

from unittest.mock import MagicMock

import app


def _fake_streamlit(monkeypatch, *, light: bool) -> MagicMock:
    """Create a mock Streamlit UI with the selected session theme."""
    fake = MagicMock()
    fake.session_state = {"light_mode": light}
    fake.columns.return_value = (MagicMock(), MagicMock(), MagicMock())
    monkeypatch.setattr(app, "st", fake)
    return fake


def test_theme_is_compact_segmented_control(monkeypatch) -> None:
    """Sun and moon remain the only two options; no huge appearance panel."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._theme_switch()
    fake.container.assert_called_once_with(key="theme_dock")
    fake.segmented_control.assert_called_once()
    args = fake.segmented_control.call_args
    assert args.kwargs["options"] == ["sun", "moon"]
    assert args.kwargs["default"] == "moon"
    assert args.kwargs["key"] == "theme_choice"
    assert args.kwargs["label_visibility"] == "collapsed"
    assert args.kwargs["format_func"]("sun") == "☀️"
    assert args.kwargs["format_func"]("moon") == "🌙"
    fake.columns.assert_not_called()
    fake.toggle.assert_not_called()


def test_theme_change_callback_updates_light_mode(monkeypatch) -> None:
    """Clicking either segment changes the actual theme state."""
    fake = _fake_streamlit(monkeypatch, light=False)
    fake.session_state["theme_choice"] = "sun"
    app._theme_changed()
    assert fake.session_state["light_mode"] is True
    fake.session_state["theme_choice"] = "moon"
    app._theme_changed()
    assert fake.session_state["light_mode"] is False


def test_theme_switch_defaults_to_selected_light_mode(monkeypatch) -> None:
    """A rerender keeps light mode selected without overwriting chat."""
    fake = _fake_streamlit(monkeypatch, light=True)
    app._theme_switch()
    assert fake.segmented_control.call_args.kwargs["default"] == "sun"
    assert fake.session_state["light_mode"] is True


def test_theme_dock_is_small_not_oversized(monkeypatch) -> None:
    """Appearance UI should be a compact pill fixed to the sidebar bottom."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._styles()
    css = fake.markdown.call_args.args[0]
    assert 'section[data-testid="stSidebar"] .st-key-theme_dock' in css
    assert "width: 132px;" in css
    assert "height: 54px;" in css
    assert "max-height: 54px;" in css
    assert "bottom: 18px;" in css
    assert ".sr-theme-label" not in css


def test_light_theme_injects_light_chat_and_sidebar_styles(monkeypatch) -> None:
    """Light mode changes the sidebar, message and background colors."""
    fake = _fake_streamlit(monkeypatch, light=True)
    app._styles()
    css = "\n".join(call.args[0] for call in fake.markdown.call_args_list)
    assert "#f6f8fd" in css
    assert "#ffffff" in css
    assert 'section[data-testid="stSidebar"]' in css
    assert 'data-testid="stChatMessage"' in css
    assert '[data-testid="stSegmentedControl"] button[aria-pressed="true"]' in css


def test_dark_theme_does_not_include_light_overrides(monkeypatch) -> None:
    """Dark mode retains the night palette during every rerender."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._styles()
    assert fake.markdown.call_count == 1
    css = fake.markdown.call_args.args[0]
    assert "#0b1020" in css
    assert "#f6f8fd" not in css


def test_native_sidebar_controls_remain_visible(monkeypatch) -> None:
    """Collapse and reopen controls must remain outside toolbar hiding."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._styles()
    css = fake.markdown.call_args.args[0]
    assert '[data-testid="stSidebarCollapsedControl"]' in css
    assert '[data-testid="stSidebarCollapseButton"]' in css
    assert '[data-testid="stToolbar"],' not in css
    assert '[data-testid="stAppDeployButton"]' in css


def test_sidebar_toggle_supports_light_theme(monkeypatch) -> None:
    """Sidebar expand/collapse buttons remain readable in light mode."""
    fake = _fake_streamlit(monkeypatch, light=True)
    app._styles()
    light_css = fake.markdown.call_args_list[-1].args[0]
    assert '[data-testid="stSidebarCollapsedControl"] button' in light_css
    assert '[data-testid="stSidebarCollapseButton"] button' in light_css
