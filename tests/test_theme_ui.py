"""Regression tests for theme placement, UI styling and sidebar controls."""

from unittest.mock import MagicMock

import app


def _fake_streamlit(monkeypatch, *, light: bool) -> MagicMock:
    """Create a mock Streamlit UI with the selected session theme."""
    fake = MagicMock()
    fake.session_state = {"light_mode": light}
    monkeypatch.setattr(app, "st", fake)
    return fake


def test_theme_switch_is_a_native_stateful_sidebar_toggle(monkeypatch) -> None:
    """A keyed Streamlit toggle reruns itself and persists the selected theme."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._theme_switch()
    fake.container.assert_called_once_with(key="theme_dock")
    fake.columns.assert_called_once_with([1, 2, 1], gap="small")
    fake.toggle.assert_called_once()
    assert fake.toggle.call_args.kwargs["key"] == "light_mode"
    assert fake.toggle.call_args.kwargs["label_visibility"] == "collapsed"
    fake.rerun.assert_not_called()
    fake.button.assert_not_called()


def test_theme_control_keeps_light_mode_selection(monkeypatch) -> None:
    """A normal rerender must not reset the user's appearance choice."""
    fake = _fake_streamlit(monkeypatch, light=True)
    app._theme_switch()
    assert fake.session_state["light_mode"] is True
    html = " ".join(call.args[0] for call in fake.markdown.call_args_list)
    assert "☀️" in html
    assert "🌙" in html


def test_theme_dock_is_bottom_left_not_top_right(monkeypatch) -> None:
    """The appearance dock must not overlap Streamlit's top-right Stop button."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._styles()
    css = fake.markdown.call_args.args[0]
    assert "section[data-testid=\"stSidebar\"] .st-key-theme_dock" in css
    assert "bottom: 16px;" in css
    assert "left: 16px;" in css
    assert ".st-key-theme_control" not in css


def test_light_theme_injects_light_chat_and_sidebar_styles(monkeypatch) -> None:
    """Light mode changes the sidebar, message and background colors."""
    fake = _fake_streamlit(monkeypatch, light=True)
    app._styles()
    css = "\n".join(call.args[0] for call in fake.markdown.call_args_list)
    assert "#f6f8fd" in css
    assert "#ffffff" in css
    assert 'section[data-testid="stSidebar"]' in css
    assert 'data-testid="stChatMessage"' in css
    assert ".sr-theme-label { color: #536587; }" in css


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
